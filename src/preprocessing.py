"""Cleaning transformers used inside the sklearn Pipeline.

Design rule: every class is a stateful sklearn transformer, so it can be
fit on training data and re-applied to raw test data without leakage. They
all accept and return pandas DataFrames so downstream FE steps can reference
columns by name.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Column dropping
# ---------------------------------------------------------------------------

class ColumnDropperByMissingness(BaseEstimator, TransformerMixin):
    """Drop columns whose NaN-fraction in the training set exceeds a threshold.

    The set of dropped columns is *learned* on `fit` and replayed on
    `transform`, so a column with 99% NaNs on test that was clean on train
    will still be kept (correct behaviour: the pipeline must not panic when
    the distribution shifts).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnDropperByMissingness expects a DataFrame")
        nan_frac = X.isna().mean()
        self.dropped_cols_ = nan_frac[nan_frac > self.threshold].index.tolist()
        self.kept_cols_    = [c for c in X.columns if c not in self.dropped_cols_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Only keep columns that exist (test set might be missing one)
        cols = [c for c in self.kept_cols_ if c in X.columns]
        return X[cols].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_cols_)


# ---------------------------------------------------------------------------
# Categorical helpers
# ---------------------------------------------------------------------------

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace categories that appear fewer than `min_count` times in train
    with the literal string '_OTHER_'. Stops one-hot from blowing up on
    high-cardinality columns like card1 (~13k unique values)."""

    def __init__(self, min_count: int = 50, columns: Optional[List[str]] = None):
        self.min_count = min_count
        self.columns   = columns

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.frequent_ = {}
        for c in cols:
            counts = X[c].astype("object").value_counts(dropna=True)
            self.frequent_[c] = set(counts[counts >= self.min_count].index)
        self.columns_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns_:
            if c not in X.columns:
                continue
            mask = X[c].isin(self.frequent_[c])
            X.loc[~mask & X[c].notna(), c] = "_OTHER_"
        return X


# ---------------------------------------------------------------------------
# Top-level configurable cleaner — drives the *_Cleaning_* MLflow runs
# ---------------------------------------------------------------------------

class IEEECleaner(BaseEstimator, TransformerMixin):
    """Configurable cleaning step that bundles the IEEE-specific tweaks.

    Parameters
    ----------
    missing_threshold : float
        Drop columns where NaN-fraction (in training) is strictly greater
        than this. Set to 1.0 to disable dropping entirely.
    amount_log : bool
        Replace TransactionAmt with log1p(TransactionAmt). Helps linear
        models, neutral for trees.
    amount_clip_q : float | None
        If set, clip TransactionAmt at this quantile (e.g. 0.999 to remove
        the long-tail outliers).
    rare_min_count : int | None
        If set, group rare categorical levels with `_OTHER_`. None disables.
    """

    def __init__(
        self,
        missing_threshold: float = 0.95,
        amount_log: bool = False,
        amount_clip_q: Optional[float] = None,
        rare_min_count: Optional[int] = 50,
    ):
        self.missing_threshold = missing_threshold
        self.amount_log        = amount_log
        self.amount_clip_q     = amount_clip_q
        self.rare_min_count    = rare_min_count

    def fit(self, X: pd.DataFrame, y=None):
        self.dropper_ = ColumnDropperByMissingness(self.missing_threshold).fit(X)
        Xk = self.dropper_.transform(X)

        if self.amount_clip_q is not None and "TransactionAmt" in Xk.columns:
            self.amt_cap_ = float(Xk["TransactionAmt"].quantile(self.amount_clip_q))
        else:
            self.amt_cap_ = None

        if self.rare_min_count is not None:
            self.rare_ = RareCategoryGrouper(self.rare_min_count).fit(Xk)
        else:
            self.rare_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xk = self.dropper_.transform(X)
        if self.amt_cap_ is not None and "TransactionAmt" in Xk.columns:
            Xk["TransactionAmt"] = Xk["TransactionAmt"].clip(upper=self.amt_cap_)
        if self.amount_log and "TransactionAmt" in Xk.columns:
            Xk["TransactionAmt"] = np.log1p(Xk["TransactionAmt"])
        if self.rare_ is not None:
            Xk = self.rare_.transform(Xk)
        return Xk
