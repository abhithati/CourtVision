"""Central configuration: filesystem paths and environment settings.

Import paths from here instead of hardcoding strings, so every module
agrees on where data and models live regardless of the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project layout (this file is src/courtvision/config.py → root is 3 levels up)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"           # untouched API/download output
INTERIM_DIR = DATA_DIR / "interim"   # cleaned, not yet feature-engineered
PROCESSED_DIR = DATA_DIR / "processed"  # model-ready feature tables
MODELS_DIR = ROOT_DIR / "models"

# Environment-driven settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'courtvision.db'}")
NBA_API_REQUEST_TIMEOUT = int(os.getenv("NBA_API_REQUEST_TIMEOUT", "30"))
