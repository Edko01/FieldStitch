"""Check GitHub releases for a newer FieldStitch and install it in place.

Each release on github.com/<UPDATE_REPO> carries:
  update.json                 {"version", "runtime", "app_zip", "sha256", "notes", "full_zip"}
  fieldstitch-app-<ver>.zip   the new "app" folder (small, used for in-place updates)
  FieldStitch-<ver>-win64.zip the full package (for fresh installs / runtime changes)
"""
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

from . import __version__, UPDATE_REPO, RUNTIME

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(APP_DIR)
UPDATE_DIR = os.path.join(ROOT_DIR, "_update")
IS_WIN = sys.platform == "win32"

state = {"phase": "idle"}  # idle | downloading | installing | restarting | error
PORT = [8765]  # set by the server once it has bound its port


def vtuple(v):
    out = []
    for part in str(v).lstrip("vV").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def installed_runtime():
    p = os.path.join(ROOT_DIR, "python", "fieldstitch_runtime.txt")
    try:
        with open(p) as f:
            return int(f.read().strip())
    except Exception:
        return RUNTIME


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "FieldStitch/" + __version__,
                                               "Accept": "application/vnd.github+json"})
    if url.startswith("http://127.0.0.1"):
        return urllib.request.urlopen(req, timeout=timeout)  # local test server only
    ctx = ssl.create_default_context()
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def check():
    base = os.environ.get("FIELDSTITCH_UPDATE_API", "https://api.github.com")
    url = "%s/repos/%s/releases/latest" % (base, UPDATE_REPO)
    try:
        with _get(url) as r:
            rel = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"ok": True, "current": __version__, "available": False,
                    "message": "No updates have been published yet."}
        return {"ok": False, "message": "GitHub answered with an error (%d). Try again later." % e.code}
    except Exception:
        return {"ok": False, "message": "Couldn't reach GitHub. Check the internet connection and try again."}
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    manifest = None
    if "update.json" in assets:
        try:
            with _get(assets["update.json"]) as r:
                manifest = json.loads(r.read().decode("utf-8"))
        except Exception:
            manifest = None
    latest = (manifest or {}).get("version") or rel.get("tag_name", "").lstrip("vV")
    available = vtuple(latest) > vtuple(__version__)
    out = {"ok": True, "current": __version__, "latest": latest, "available": available,
           "notes": (manifest or {}).get("notes") or rel.get("body") or "",
           "page": rel.get("html_url"), "published": rel.get("published_at")}
    if available:
        need_rt = int((manifest or {}).get("runtime", RUNTIME))
        app_zip = (manifest or {}).get("app_zip")
        if not manifest or not app_zip or app_zip not in assets:
            out["needs_full"] = True
        elif need_rt > installed_runtime():
            out["needs_full"] = True
        else:
            out["needs_full"] = False
            out["_download"] = {"url": assets[app_zip], "sha256": manifest.get("sha256", "")}
        full = (manifest or {}).get("full_zip")
        if full and full in assets:
            out["full_url"] = assets[full]
    return out


def install(busy_reason=None):
    """Download and stage the update, then restart into it (runs in a thread)."""
    if busy_reason:
        raise RuntimeError(busy_reason)
    if state.get("phase") in ("downloading", "installing", "restarting"):
        raise RuntimeError("An update is already being installed.")
    info = check()
    if not info.get("ok"):
        raise RuntimeError(info.get("message"))
    if not info.get("available"):
        raise RuntimeError("FieldStitch is already up to date.")
    if info.get("needs_full"):
        raise RuntimeError("This update needs the full download. Use the download link instead.")
    state.clear()
    state.update(phase="downloading", frac=0.0, version=info["latest"])
    threading.Thread(target=_install_worker, args=(info,), daemon=True).start()
    return info


def _install_worker(info):
    try:
        if os.path.exists(UPDATE_DIR):
            shutil.rmtree(UPDATE_DIR, ignore_errors=True)
        os.makedirs(UPDATE_DIR, exist_ok=True)
        zpath = os.path.join(UPDATE_DIR, "app.zip")
        h = hashlib.sha256()
        with _get(info["_download"]["url"], timeout=60) as r, open(zpath, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if total:
                    state["frac"] = got / total
        want = (info["_download"].get("sha256") or "").lower()
        if want and h.hexdigest() != want:
            raise RuntimeError("The download was damaged (checksum mismatch). Try again.")
        state.update(phase="installing")
        new_dir = os.path.join(UPDATE_DIR, "app.new")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(UPDATE_DIR)
        # the zip contains a top-level "app" folder
        src = os.path.join(UPDATE_DIR, "app")
        if not (os.path.isfile(os.path.join(src, "main.py"))
                and os.path.isfile(os.path.join(src, "fieldstitch", "__init__.py"))):
            raise RuntimeError("The update package is incomplete.")
        os.replace(src, new_dir)
        swap = os.path.join(UPDATE_DIR, "swap.py")
        with open(swap, "w", encoding="utf-8") as f:
            f.write(SWAP_SCRIPT)
        state.update(phase="restarting")
        flags = 0
        if IS_WIN:
            flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
        subprocess.Popen([sys.executable, swap, ROOT_DIR, str(os.getpid()), str(PORT[0])], cwd=ROOT_DIR,
                         creationflags=flags, close_fds=True,
                         start_new_session=not IS_WIN)
        time.sleep(1.5)  # let the browser see the "restarting" state
        os._exit(0)
    except Exception as e:
        state.update(phase="error", message=str(e))


SWAP_SCRIPT = r'''
import os, shutil, subprocess, sys, time
root, pid, PORT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
app = os.path.join(root, "app")
new = os.path.join(root, "_update", "app.new")
old = os.path.join(root, "_update", "app.old")
log = open(os.path.join(root, "_update", "swap.log"), "a")
def say(m):
    log.write(time.strftime("%H:%M:%S ") + m + "\n"); log.flush()
time.sleep(2.0)
ok = False
for attempt in range(60):
    try:
        if os.path.exists(old):
            shutil.rmtree(old, ignore_errors=True)
        os.replace(app, old)
        try:
            os.replace(new, app)
        except Exception:
            os.replace(old, app)
            raise
        ok = True
        break
    except Exception as e:
        say("waiting for files to be released: %s" % e)
        time.sleep(1.0)
say("swap " + ("done" if ok else "FAILED, kept the previous version"))
py = os.path.join(root, "python", "python.exe")
if not os.path.exists(py):
    py = sys.executable
cmd = [py, os.path.join(root, "app", "main.py"), "--no-browser"]

def alive(port):
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:%d/api/status" % port, timeout=2).read()
        return True
    except Exception:
        return False

try:
    if sys.platform == "win32":
        subprocess.Popen(cmd, cwd=root, creationflags=0x00000010)  # CREATE_NEW_CONSOLE
    else:
        subprocess.Popen(cmd, cwd=root, start_new_session=True)
    say("restarted")
except Exception as e:
    say("restart failed: %s" % e)
up = False
for _ in range(25):
    time.sleep(1)
    if alive(PORT):
        up = True
        break
if not up:
    # fallback: run without a console window, output to a log file
    say("no answer on port %d, starting again without a console" % PORT)
    out = open(os.path.join(root, "_update", "restart.log"), "w")
    flags = 0x08000000 if sys.platform == "win32" else 0
    subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                     creationflags=flags)
if ok:
    time.sleep(3)
    shutil.rmtree(old, ignore_errors=True)
'''
