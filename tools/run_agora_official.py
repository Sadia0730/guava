#!/usr/bin/env python
"""Run the official AGORA evaluation (github.com/pixelite1201/agora_evaluation, an unmodified checkout) on a folder of
prediction pickles written by tools/eval_agora.py (phase3_dnt/PREREG.md, "Pose table ... AGORA").
The predictions carry joints and vertices, so no body model is needed: `load_model` is replaced by a stub (the official
code otherwise loads six SMPL-X models it would never use) and `smplx` is stubbed if it is not installed.
Usage: python tools/run_agora_official.py <agora_evaluation checkout> <evaluate_agora arguments ...>"""
import sys
import types

repo = sys.argv[1]
sys.path.insert(0, repo)
try:
    import smplx  # noqa: F401
except ImportError:
    sys.modules["smplx"] = types.ModuleType("smplx")
import agora_evaluation.get_joints_verts_from_dataframe as g  # noqa: E402

g.load_model = lambda args: (None,) * 6
from agora_evaluation.evaluate_agora import run_evaluation  # noqa: E402

run_evaluation(sys.argv[2:])
