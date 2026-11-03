"""Package manifest: run live_sprint_guavafb.py under an audit hook and record every file it opens (outside the Python
installation) and every third-party module it imports (with versions). Usage: python audit_run.py OUT.json -- <app args>"""
import json
import os
import runpy
import sys
from importlib import metadata
from pathlib import Path

out = Path(sys.argv[1]); args = sys.argv[sys.argv.index("--") + 1:]
APP = Path(__file__).with_name("live_sprint_guavafb.py")
opened = set(); prefixes = (sys.prefix, sys.base_prefix, "/proc", "/sys", "/dev", "/etc", "/usr")


def hook(event, a):
    if event == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
        p = os.path.abspath(os.fsdecode(a[0]))
        if not p.startswith(prefixes) and os.path.isfile(p): opened.add(p)


sys.addaudithook(hook)
sys.argv = [str(APP)] + args
try:
    runpy.run_path(str(APP), run_name="__main__")
finally:
    mods = {}
    for name, m in list(sys.modules.items()):
        if "." in name or not getattr(m, "__file__", None): continue
        try: mods[name] = metadata.version(name)
        except Exception: mods[name] = None                                                         # noqa: BLE001
    files = sorted(f for f in opened if os.path.isfile(f))
    out.write_text(json.dumps({"files": files, "sizes": {f: os.path.getsize(f) for f in files}, "modules": mods,
                               "python": sys.version, "args": args}, indent=1))
    print(f"audit: {len(files)} files, {len(mods)} modules -> {out}", flush=True)
