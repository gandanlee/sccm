"""SCCM — Spherically Consistent Coarse Matching for ERP dense feature correspondence.

Paper-path subset of the RoMa codebase (Edstedt et al., CVPR 2024), extended with
SPA (Spherical Positional Attention) and AAC (Area-Aware Covisibility).
"""
import os

__version__ = "1.0.0"

DEBUG_MODE = False
RANK = int(os.environ.get("RANK", default=0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1
