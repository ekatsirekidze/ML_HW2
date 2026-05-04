"""Loaders for the IEEE-CIS Fraud Detection CSVs.

These functions auto-detect whether the data lives at the Kaggle mount point
(`/kaggle/input/ieee-fraud-detection`) or in the local working directory, so
the same notebook code works in both places.

The Kaggle test-identity export uses dashes in column names (e.g. `id-01`)
while the train file uses underscores (`id_01`). We normalise to underscores
on load.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


KAGGLE_PATH = "/kaggle/input/ieee-fraud-detection"
LOCAL_PATH  = "."


def _detect_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    for p in (KAGGLE_PATH, LOCAL_PATH):
        if os.path.exists(os.path.join(p, "train_transaction.csv")):
            return p
    raise FileNotFoundError(
        "Could not locate IEEE-CIS CSVs. Looked at: "
        f"{KAGGLE_PATH!r}, {LOCAL_PATH!r}. "
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
