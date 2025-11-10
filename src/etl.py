from __future__ import annotations
import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def clean_data(raw_path: str = "data/raw/synthetic_dataset.csv",
               out_path: str = "data/processed/clean.csv") -> str:
    """
    Load raw CSV/Excel, standardize columns/dtypes, handle missing values,
    remove duplicates by subject_id, and write a clean CSV.
    """
    # Load CSV or Excel by extension
    p = Path(raw_path)
    if p.suffix.lower() == ".xlsx":
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
                   .str.lower()
                   .str.replace(" ", "_")
    )

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Ensure types
    cat_cols = ["group", "site", "subject_id"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # Remove duplicate subjects keeping the first
    if "subject_id" in df.columns:
        df = df.drop_duplicates(subset=["subject_id"], keep="first")

    # Simple missing handling example: impute numeric with column mean
    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    # Derive a standardized score from score1 (z-score like)
    if "score1" in df.columns:
        df["score1_z"] = (df["score1"] - df["score1"].mean()) / df["score1"].std(ddof=0)

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

def load_cleaned(path: str = "data/processed/clean.csv") -> pd.DataFrame:
    """Convenience loader for downstream steps."""
    return pd.read_csv(path)
