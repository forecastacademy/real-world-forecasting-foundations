"""
utils/paths.py

Centralized path management for the Forecast Academy course repo.
Ensures that all notebooks load data consistently, regardless of
where the user runs them (Windows/Mac/Linux, root/notebooks folder).

Usage:
    from utils.paths import DATA_DIR, ARTIFACTS_DIR, RAW_DIR, load_parquet
"""

from pathlib import Path
import os
import pandas as pd
import json

# ---------------------------------------------------------------------------
# 1. Locate the repo root directory
# ---------------------------------------------------------------------------

# This file lives in: repo_root/utils/paths.py
# So repo_root = parent of the parent directory
ROOT_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 2. Define key directories
# ---------------------------------------------------------------------------

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DOCS_DIR = ROOT_DIR / "docs"
SCRIPTS_DIR = ROOT_DIR / "scripts"


# ---------------------------------------------------------------------------
# 3. Directory creation helper
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create the full folder structure if it does not exist."""
    for d in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        INTERIM_DIR,
        ARTIFACTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# Create dirs on import (safe & fast)
ensure_dirs()


# ---------------------------------------------------------------------------
# 4. File helper functions
# ---------------------------------------------------------------------------

def load_parquet(path: str | Path):
    """Load a parquet file using pandas."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path):
    """Save dataframe to parquet (creates directories if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_csv(path: str | Path):
    """Load a CSV file."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path):
    """Save dataframe to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_json(path: str | Path):
    """Load a JSON file."""
    return json.loads(Path(path).read_text())


def save_json(obj, path: str | Path):
    """Save JSON serializable object to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# ---------------------------------------------------------------------------
# 5. Friendly shortcuts for common course assets
# ---------------------------------------------------------------------------

def diagnostics_path():
    """Return the path to the diagnostics parquet from Module 1.8."""
    return ARTIFACTS_DIR / "1_8_diagnostics.parquet"


def summary_path():
    """Return the path to the GenAI summary (Module 1.9)."""
    return ARTIFACTS_DIR / "1_9_summary.txt"


def weekly_data_path():
    """Return the final weekly dataset (Module 1.10)."""
    return PROCESSED_DIR / "m5_weekly.parquet"


# ---------------------------------------------------------------------------
# 6. Human-friendly listing (optional)
# ---------------------------------------------------------------------------

def describe():
    """Print all key directories for debugging."""
    print("ROOT_DIR:", ROOT_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("INTERIM_DIR:", INTERIM_DIR)
    print("ARTIFACTS_DIR:", ARTIFACTS_DIR)
    print("NOTEBOOKS_DIR:", NOTEBOOKS_DIR)
    print("DOCS_DIR:", DOCS_DIR)
    print("SCRIPTS_DIR:", SCRIPTS_DIR)
