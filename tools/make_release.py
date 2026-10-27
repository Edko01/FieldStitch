"""Build release assets for FieldStitch: app zip + update.json (+ optional full zip)."""
import hashlib, json, os, sys, zipfile
app_dir, out_dir, notes = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
full_zip = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, app_dir)
import importlib.util
spec = importlib.util.spec_from_file_location("fsinit", os.path.join(app_dir, "fieldstitch", "__init__.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ver = m.__version__
os.makedirs(out_dir, exist_ok=True)
zname = "fieldstitch-app-%s.zip" % ver
zp = os.path.join(out_dir, zname)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(app_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.join("app", os.path.relpath(full, app_dir)))
sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()
man = {"version": ver, "runtime": m.RUNTIME, "app_zip": zname, "sha256": sha, "notes": notes}
if full_zip:
    man["full_zip"] = os.path.basename(full_zip)
json.dump(man, open(os.path.join(out_dir, "update.json"), "w"), indent=1)
print(ver, zname, os.path.getsize(zp), sha)
