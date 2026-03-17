"""
Backward-compatibility shim.

The plot functions now live in utils/graficas.py.
This file re-exports them so that legacy scripts (e.g. T_UAV_CBF_baseline.py)
still work with ``from graficas import ...``.
"""

from utils.graficas import *  # noqa: F401,F403
