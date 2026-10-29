"""Open PlaneSite in its own app window (Microsoft Edge / Chrome app mode), and find a running instance."""
import json
import os
import subprocess
import sys
import urllib.request

IS_WIN = sys.platform == "win32"
PORTS = range(8765, 8795)


def data_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "PlaneSite")
    os.makedirs(d, exist_ok=True)
    return d


def find_running():
    """URL of an already running PlaneSite server, or None."""
    for p in PORTS:
        url = "http://127.0.0.1:%d/" % p
        try:
            with urllib.request.urlopen(url + "api/status", timeout=0.4) as r:
                if "PlaneSite" in (r.headers.get("Server") or ""):
                    json.loads(r.read().decode("utf-8"))
                    return url
        except Exception:
            continue
    return None


def _browser_candidates():
    c = []
    if IS_WIN:
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for exe in ("msedge.exe", "chrome.exe"):
                    try:
                        k = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\%s" % exe)
                        c.append(winreg.QueryValue(k, None))
                    except OSError:
                        pass
        except Exception:
            pass
        for env in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                c.append(os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"))
                c.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
    else:
        for exe in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
            for d in os.environ.get("PATH", "").split(os.pathsep):
                c.append(os.path.join(d, exe))
    seen = []
    for p in c:
        if p and os.path.isfile(p) and p not in seen:
            seen.append(p)
    return seen


# The window is a browser page; keep the browser from pausing or slowing it when it is covered,
# in the background or the computer is busy processing (Edge's efficiency mode / tab freezing).
WINDOW_FLAGS = [
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=Translate,CalculateNativeWinOcclusion,IntensiveWakeUpThrottling",
]


def open_window(url):
    """Open url as a standalone app window. Returns the browser Popen (or None if a normal tab was used)."""
    profile = os.path.join(data_dir(), "window")
    for exe in _browser_candidates():
        try:
            return subprocess.Popen([exe, "--app=" + url, "--user-data-dir=" + profile, "--window-size=1440,900",
                                     "--no-first-run", "--no-default-browser-check"] + WINDOW_FLAGS,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue
    import webbrowser
    webbrowser.open(url)
    return None
