"""Shared fixtures.

LightGBM on the tiny datasets used here is far slower with its default thread pool than
with one thread, so pin it before any test imports the library.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
