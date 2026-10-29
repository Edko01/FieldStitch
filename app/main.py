"""PlaneSite launcher.

Starts the local processing server (or reuses one that is already running) and opens the
app window. Run with pythonw.exe for no console window.
  --no-window   start the server only (used after an update restart)
  --quit        close a running PlaneSite (used by the installer)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    from planesite import appwindow
    no_window = "--no-window" in sys.argv or "--no-browser" in sys.argv
    if sys.stdout is None or os.path.basename(sys.executable).lower() in ("pythonw.exe", "planesitew.exe"):
        logdir = os.path.join(appwindow.data_dir(), "logs")
        os.makedirs(logdir, exist_ok=True)
        log = open(os.path.join(logdir, "planesite.log"), "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = log
    elif sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("PlaneSite")
        except Exception:
            pass
    if "--quit" in sys.argv:  # used by the installer / uninstaller
        import json, urllib.request
        url = appwindow.find_running()
        if url:
            try:
                req = urllib.request.Request(url + "api/quit", data=json.dumps({"force": True}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).read()
            except Exception:
                pass
        sys.exit(0)
    running = appwindow.find_running()
    if running:
        if not no_window:
            appwindow.open_window(running)
        sys.exit(0)
    from planesite.server import serve
    serve(open_browser=not no_window)
