"""Run one of GaussianAvatar's official preprocessing scripts unchanged except for its hard-coded path lines
(the README tells the user to edit them: "remember to change the path").

  python ga_official.py <repo>/scripts/<script>.py "<old line>" "<new line>" ["<old line>" "<new line>" ...]

Each <old line> must occur exactly once in the script. The script runs as __main__ with cwd = its own folder and
that folder first on sys.path, exactly like `cd scripts && python <script>.py`.
"""
import os
import sys


def main():
    script = os.path.abspath(sys.argv[1])
    subs = sys.argv[2:]
    assert len(subs) % 2 == 0, "pairs of <old line> <new line>"
    src = open(script).read()
    for old, new in zip(subs[0::2], subs[1::2]):
        n = src.count(old)
        assert n == 1, f"{old!r} occurs {n} times in {script}"
        src = src.replace(old, new)
    sdir = os.path.dirname(script)
    os.chdir(sdir)
    sys.path.insert(0, sdir)
    sys.argv = [script]
    exec(compile(src, script, "exec"), {"__name__": "__main__", "__file__": script})


if __name__ == "__main__":
    main()
