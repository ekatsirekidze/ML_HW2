"""Loaders for the IEEE-CIS Fraud Detection CSVs.

The path-detection logic checks several known Kaggle mount points (different
notebook flavours mount the data at different paths), the local working
directory, and finally falls back to scanning `/kaggle/input/*` for any folder
containing `train_transaction.csv`. This makes the same notebook code work in
all of:

  * Kaggle "Competition" notebooks  ->  /kaggle/input/competitions/ieee-fraud-detection
  * Kaggle notebooks where the dataset was attached manually
                                     ->  /kaggle/input/ieee-fraud-detection
  * Local Jupyter where the CSVs sit beside the notebook
                                     ->  .

The Kaggle test-identity export uses dashes in column names (e.g. `id-01`)
while the train file uses underscores (`id_01`). We normalise to underscores
on load.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import pandas as pd


# Candidate locations, in priority order. First one whose `train_transaction.csv`
# exists wins. Add new mount points here without touching the rest of the code.
KAGGLE_PATH_CANDIDATES = [
    "/kaggle/input/competitions/ieee-fraud-detection",
    "/kaggle/input/ieee-fraud-detection",
]
LOCAL_PATH = "."


def _detect_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit

    for p in (*KAGGLE_PATH_CANDIDATES, LOCAL_PATH):
        if os.path.exists(os.path.join(p, "train_transaction.csv")):
            return p

    # Final fallback: scan /kaggle/input/* for any folder containing the file.
    for hit in glob.glob("/kaggle/input/*/train_transaction.csv"):
        return os.path.dirname(hit)
    for hit in glob.glob("/kaggle/input/*/*/train_transaction.csv"):
        return os.path.dirname(hit)

    raise FileNotFoundError(
        "Could not locate IEEE-CIS CSVs. Looked at: "
        f"{KAGGLE_PATH_CANDIDATES + [LOCAL_PATH]!r}. "
        "Pass `path=` explicitly or attach the dataset on Kaggle."
    )


def _normalise_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace("-", "_", regex=False)
    return df


def load_train(
    path: Optional[str] = None,
    *,
    nrows: Optional[int] = None,
    merge: bool = True,
) -> pd.DataFrame:
    """Load training data. If `merge=True` (default) returns the
    transaction+identity LEFT JOIN; otherwise returns the transaction frame
    only."""
    p = _detect_path(path)
    tx = pd.read_csv(os.path.join(p, "train_transaction.csv"), nrows=nrows)
    if not merge:
        return tx
    idn = pd.read_csv(os.path.join(p, "train_identity.csv"))
    idn = _normalise_id_columns(idn)
    return tx.merge(idn, on="TransactionID", how="left")


def load_test(
    path: Optional[str] = None,
    *,
    merge: bool = True,
) -> pd.DataFrame:
    """Load test data. Identity columns are normalised to underscores."""
    p = _detect_path(path)
    tx = pd.read_csv(os.path.join(p, "test_transaction.csv"))
    if not merge:
        return tx
    idn = pd.read_csv(os.path.join(p, "test_identity.csv"))
    idn = _normalise_id_columns(idn)
    return tx.merge(idn, on="TransactionID", how="left")


def load_sample_submission(path: Optional[str] = None) -> pd.DataFrame:
    p = _detect_path(path)
    return pd.read_csv(os.path.join(p, "sample_submission.csv"))


def downcast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory by ~60% for IEEE data: float64->float32, int64->int32.
    Leaves object/category columns untouched. Returns a NEW DataFrame."""
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype("float32")
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = out[col].astype("int32")
    return out
