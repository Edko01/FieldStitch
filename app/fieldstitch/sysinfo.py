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
        res = []
        for g in data:
            ram = g.get("AdapterRAM")
            res.append({"name": g.get("Name"), "driver": g.get("DriverVersion"),
                        "memory_gb": round(ram / 1024 ** 3, 1) if isinstance(ram, (int, float)) and ram > 0 else None})
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
