"""Publish an in-app update into the repo's updates/ folder.

usage: python tools/make_release.py "release notes" [full_package_url]
Builds updates/fieldstitch-app-<version>.zip from app/ and writes updates/latest.json.
Commit and push afterwards; installed copies pick it up from "Check for updates".
"""
import datetime, glob, hashlib, importlib.util, json, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
OUT = os.path.join(ROOT, "updates")
notes = sys.argv[1] if len(sys.argv) > 1 else ""
full_url = sys.argv[2] if len(sys.argv) > 2 else ""
spec = importlib.util.spec_from_file_location("fsinit", os.path.join(APP, "fieldstitch", "__init__.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ver = m.__version__
os.makedirs(OUT, exist_ok=True)
for old in glob.glob(os.path.join(OUT, "fieldstitch-app-*.zip")):  # keep the repo small
    os.remove(old)
zname = "fieldstitch-app-%s.zip" % ver
zp = os.path.join(OUT, zname)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(APP):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            full = os.path.join(root, f)
            z.write(full, os.path.join("app", os.path.relpath(full, APP)))
sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()
man = {"version": ver, "runtime": m.RUNTIME, "app_zip": "updates/" + zname, "sha256": sha,
       "notes": notes, "date": datetime.date.today().isoformat()}
if full_url:
    man["full_url"] = full_url
with open(os.path.join(OUT, "latest.json"), "w") as f:
    json.dump(man, f, indent=1)
print("FieldStitch", ver, "->", zname, os.path.getsize(zp), "bytes, sha256", sha)
