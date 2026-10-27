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
