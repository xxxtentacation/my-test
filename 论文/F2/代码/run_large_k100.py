# -*- coding: utf-8 -*-
"""
The large scale of the study, K = 100, on its own.

This is ``run_large.py`` restricted to its K=100 group: the sixteen instance
sizes 100, 120, ..., 400 crossed with the two processing-time ranges, twenty
instances per configuration.  Everything else -- the methods, the parameters,
the records, the report -- is the driver's, so the two cannot drift apart.

Every command-line argument except ``--scale`` is passed through; ``--out``
defaults to a file of this group's own, so that the two groups can be run side
by side without overwriting each other's latest results.

Usage:
    python run_large_k100.py                 # the K=100 group
    python run_large_k100.py --list          # what would be run
    python run_large_k100.py --algos ga      # the GA alone
    python run_large_k100.py --instances 2   # a quick check
    python run_large_k100.py --out results.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

from run_large import main

HERE = Path(__file__).resolve().parent
GROUP = "large100"
DEFAULT_OUT = HERE / "记录" / "结果_large_k100_最新.txt"


def _argv():
    """The driver's command line, fixed to this group and its output file."""
    argv = sys.argv[1:]
    if "--out" not in argv and not any(a.startswith("--out=") for a in argv):
        argv = ["--out", str(DEFAULT_OUT)] + argv
    return ["--scale", GROUP] + argv


if __name__ == "__main__":
    sys.exit(main(_argv()))
