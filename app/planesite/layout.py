"""Where things live inside a project folder.

    <project>/
        project.json            the project (also how a folder is recognised as a project)
        exports/                everything the user sends out: GeoTIFF, XAG tiles, Boundaries, Pilot 2
        PlaneSite data/         PlaneSite's own files; the user never needs to open it
            processing/         the processing result shown on the map (orthomosaic, surface, report)
            work/               temporary processing files (features, thumbnails)
            basemap/            cached satellite imagery
            areas.json, log.txt, progress.json, process_console.txt

Before 2.0 those internal files sat directly in the project folder (with "outputs" for
"processing"). Every path here falls back to the old place while a file is still there, and
migrate() moves an old project over when it's opened.
"""
import os

DATA = "PlaneSite data"
EXPORTS = "exports"

# new location (inside DATA) -> where it was before 2.0 (in the project folder)
_ITEMS = {
    "processing": "outputs",
    "work": "work",
    "basemap": "basemap",
    "areas.json": "areas.json",
    "log.txt": "log.txt",
    "progress.json": "progress.json",
    "process_console.txt": "process_console.txt",
}


def _path(d, name):
    new = os.path.join(d, DATA, name)
    old = os.path.join(d, _ITEMS[name])
    if not os.path.exists(new) and os.path.exists(old):
        return old
    return new


def data_dir(d):
    return os.path.join(d, DATA)


def processing_dir(d):
    return _path(d, "processing")


def work_dir(d):
    return _path(d, "work")


def basemap_dir(d):
    return _path(d, "basemap")


def areas_file(d):
    return _path(d, "areas.json")


def log_file(d):
    return _path(d, "log.txt")


def progress_file(d):
    return _path(d, "progress.json")


def console_file(d):
    return _path(d, "process_console.txt")


def exports_dir(d, *parts):
    return os.path.join(d, EXPORTS, *parts)


def migrate(d):
    """Move a pre-2.0 project's internal files into 'PlaneSite data'. Renames only (instant, same
    drive). Anything that can't be moved right now (a file open elsewhere) stays where it is and is
    still found there. Returns the names moved."""
    moved = []
    if not d or not os.path.isdir(d):
        return moved
    for new, old in _ITEMS.items():
        src, dst = os.path.join(d, old), os.path.join(d, DATA, new)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.makedirs(os.path.join(d, DATA), exist_ok=True)
                os.replace(src, dst)
                moved.append(old)
            except OSError:
                pass
    return moved
