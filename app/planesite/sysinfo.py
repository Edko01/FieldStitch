import ctypes
import os
import sys


def total_ram_bytes():
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MS()
            st.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return 8 * 1024 ** 3


def cpu_count():
    return max(1, os.cpu_count() or 1)


def worker_count(per_worker_bytes, reserve_bytes=2 * 1024 ** 3, max_workers=None):
    ram = total_ram_bytes()
    n = cpu_count() - 1 if cpu_count() > 2 else cpu_count()
    by_ram = max(1, int((ram - reserve_bytes) // max(per_worker_bytes, 1)))
    n = max(1, min(n, by_ram))
    if max_workers:
        n = min(n, max_workers)
    return n


# ------------------------------------------------------------------ diagnostics
def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or "unknown"


def gpus():
    """List of graphics adapters (name, memory, driver). Best effort, Windows only."""
    if sys.platform != "win32":
        return []
    import json
    import subprocess
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, creationflags=0x08000000).stdout
        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
        # Windows reports AdapterRAM as a 32-bit number, so every card with 4 GB or more shows 4 GB;
        # NVIDIA's own tool gives the real size
        nv = {}
        try:
            import shutil
            exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
            o = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10, creationflags=0x08000000).stdout
            for line in o.splitlines():
                nm, _, mb = line.rpartition(",")
                if nm.strip() and mb.strip().isdigit():
                    nv[nm.strip()] = round(int(mb) / 1024, 1)
        except Exception:
            pass
        res = []
        for g in data:
            ram = g.get("AdapterRAM")
            gb = round(ram / 1024 ** 3, 1) if isinstance(ram, (int, float)) and ram > 0 else None
            if gb is not None and gb >= 3.99:
                gb = None                    # capped value: unknown, at least 4 GB
            name = g.get("Name")
            gb = nv.get(name, gb)
            res.append({"name": name, "driver": g.get("DriverVersion"), "memory_gb": gb})
        return res
    except Exception:
        return []


def drive_kind(path):
    """'local', 'removable', 'network' or 'unknown' for the drive holding path."""
    try:
        if path.startswith("\\\\"):
            return "network"
        if sys.platform == "win32":
            import ctypes
            root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
            t = ctypes.windll.kernel32.GetDriveTypeW(root)
            return {2: "removable", 3: "local", 4: "network", 5: "cd", 6: "ramdisk"}.get(t, "unknown")
        return "local"
    except Exception:
        return "unknown"


def free_gb(path):
    try:
        import shutil
        return round(shutil.disk_usage(path).free / 1024 ** 3, 1)
    except Exception:
        return None


def describe(project_dir=None, photo_dirs=()):
    import platform
    info = {"os": platform.platform(), "cpu": cpu_name(), "logical_cpus": cpu_count(),
            "ram_gb": round(total_ram_bytes() / 1024 ** 3, 1), "python": platform.python_version(),
            "gpus": gpus()}
    if project_dir:
        info["project_drive"] = drive_kind(project_dir)
        info["project_free_gb"] = free_gb(project_dir)
    kinds = sorted(set(drive_kind(d) for d in photo_dirs if d))
    if kinds:
        info["photo_drives"] = kinds
    return info


FULL_SPEED_ERR = [""]


def full_speed():
    """Ask Windows not to treat this (windowless, background) process as low priority work.

    Windows 11 applies "power throttling" (EcoQoS / efficiency mode) to background processes:
    on Intel hybrid CPUs their threads are steered to the efficiency cores at low clock speeds.
    Processing workers have no window, so without this they can run far below the CPU's
    capacity. Safe to call anywhere; does nothing on other systems."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        h = k32.GetCurrentProcess()
        # ProcessPowerThrottling = 4; control EXECUTION_SPEED (1) and IGNORE_TIMER_RESOLUTION (4); state 0 = off
        st = _PPTS(1, 0x1 | 0x4, 0)
        ok = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
        if not ok:
            # older builds reject IGNORE_TIMER_RESOLUTION; retry with execution speed only
            err1 = ctypes.get_last_error()
            st = _PPTS(1, 0x1, 0)
            ok = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
            if not ok:
                FULL_SPEED_ERR[0] = "Windows error %d / %d" % (err1, ctypes.get_last_error())
        return ok
    except Exception as e:  # noqa
        FULL_SPEED_ERR[0] = "%s: %s" % (type(e).__name__, e)
        return False


def cpu_times():
    """(idle, total) CPU time counters for the whole machine, for measuring how busy it is."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            i, k, u = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
            f = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa
            return f(i), f(k) + f(u)   # kernel time includes idle time
        with open("/proc/stat") as fh:
            v = [int(x) for x in fh.readline().split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return idle, sum(v)
    except Exception:
        return None


def raise_priority():
    """Above-normal priority: lets a process that feeds the GPU keep up while the CPU is saturated."""
    if sys.platform != "win32":
        try:
            os.nice(-2)
        except Exception:
            pass
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), 0x8000)  # ABOVE_NORMAL
    except Exception:
        pass


class GpuMonitor:
    """Samples NVIDIA GPU utilisation with nvidia-smi (ships with the driver) in a thread."""

    def __init__(self, every=2.0):
        import shutil
        import threading
        self.exe = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe"
                                                    if sys.platform == "win32" else None)
        self.samples = []
        self.every = every
        self._stop = threading.Event()
        self._t = None
        if self.exe and os.path.exists(self.exe):
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()

    def _run(self):
        import subprocess
        flags = 0x08000000 if sys.platform == "win32" else 0
        while not self._stop.wait(self.every):
            try:
                out = subprocess.run([self.exe, "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5, creationflags=flags).stdout.strip().splitlines()
                if out:
                    u, m = [float(x) for x in out[0].split(",")[:2]]
                    self.samples.append((u, m))
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=6)
        if not self.samples:
            return None
        u = [s[0] for s in self.samples]
        return {"avg_util": sum(u) / len(u), "max_util": max(u), "max_mem_mb": max(s[1] for s in self.samples),
                "n": len(u)}
