"""PlaneSite - fast offline orthomosaic generation from drone photos."""
__version__ = "1.6.11"
APP_NAME = "PlaneSite"
UPDATE_REPO = "Edko01/PlaneSite"   # GitHub repo that hosts updates/latest.json
RUNTIME = 3                          # version of the bundled python/ folder this app needs


def migrate_from_fieldstitch():
    """One-time carry-over of settings and timing calibration from the app's old name.

    Projects are not carried over: the old "FieldStitch Projects" folder is left alone, and
    recent-project and default-folder entries that point into it are dropped.
    """
    import json, os
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    old, new = os.path.join(base, "FieldStitch"), os.path.join(base, "PlaneSite")
    if not os.path.isdir(old) or os.path.exists(os.path.join(new, ".migrated")):
        return
    try:
        os.makedirs(new, exist_ok=True)
        cal = os.path.join(old, "timing_calibration.json")
        if os.path.isfile(cal) and not os.path.exists(os.path.join(new, "timing_calibration.json")):
            with open(cal, "rb") as f:
                data = f.read()
            with open(os.path.join(new, "timing_calibration.json"), "wb") as f:
                f.write(data)
        cfg_old = os.path.join(old, "config.json")
        if os.path.isfile(cfg_old) and not os.path.exists(os.path.join(new, "config.json")):
            with open(cfg_old, encoding="utf-8") as f:
                cfg = json.load(f)
            def stale(path):
                return "fieldstitch projects" in os.path.normcase(str(path or "")).replace("\\", "/")
            cfg["recent"] = [r for r in cfg.get("recent", []) if not stale(r)]
            if stale(cfg.get("projects_dir")):
                cfg.pop("projects_dir", None)
            with open(os.path.join(new, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1)
        open(os.path.join(new, ".migrated"), "w").close()
    except Exception:
        pass
