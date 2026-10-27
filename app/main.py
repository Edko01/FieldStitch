"""FieldStitch launcher: starts the local server and opens the browser."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("FieldStitch")
        except Exception:
            pass
    from fieldstitch.server import serve
    no_browser = "--no-browser" in sys.argv
    serve(open_browser=not no_browser)
