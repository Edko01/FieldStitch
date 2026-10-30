"""Secrets kept for this Windows user in Windows Credential Manager (a private file elsewhere,
for development on other systems). Never in the project or the settings file."""
import os
import sys


def _file(name):
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "PlaneSite", "secret_" + name.replace("/", "_"))


def save(name, secret, comment="PlaneSite"):
    """Store (or with an empty secret, delete) the secret under PlaneSite/<name>."""
    if sys.platform == "win32":
        from .pilot2 import _win_cred_write
        _win_cred_write("PlaneSite/" + name, secret or "", comment)
        return
    p = _file(name)
    if not secret:
        if os.path.exists(p):
            os.remove(p)
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secret)


def load(name):
    if sys.platform == "win32":
        from .pilot2 import _win_cred_read
        return _win_cred_read("PlaneSite/" + name)
    try:
        with open(_file(name)) as f:
            return f.read() or None
    except OSError:
        return None
