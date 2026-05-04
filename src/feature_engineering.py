"""Feature-engineering transformers for IEEE-CIS Fraud Detection.

Each transformer is independently toggleable via flags on
`IEEEFeatureEngineer`, so MLflow runs can compare strategies cleanly:

    fe_v1 = IEEEFeatureEngineer(use_datetime=True,  use_email=False, ...)
    fe_v2 = IEEEFeatureEngineer(use_datetime=True,  use_email=True,  ...)
    fe_v3 = IEEEFeatureEngineer(use_datetime=True,  use_email=True,  use_freq=True, ...)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Calendar features from TransactionDT
# ---------------------------------------------------------------------------

class DatetimeFromTransactionDT(BaseEstimator, TransformerMixin):
    """Derive hour/dow/day/month/weekofyear from `TransactionDT`.

    `TransactionDT` is "seconds since some reference date". The reference
    used by Vesta is 2017-12-01 (well-known on Kaggle). Calendar features
    are reasonably invariant to which reference you pick, since the data
    only spans ~6 months, so this is a safe choice.
    """

    REF = pd.Timestamp("2017-12-01")

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "TransactionDT" not in X.columns:
            return X
        dt = self.REF + pd.to_timedelta(X["TransactionDT"], unit="s")
        X["TX_hour"]       = dt.dt.hour.astype("int16")
        X["TX_dow"]        = dt.dt.dayofweek.astype("int8")
        X["TX_day"]        = dt.dt.day.astype("int8")
        X["TX_month"]      = dt.dt.month.astype("int8")
        X["TX_weekofyear"] = dt.dt.isocalendar().week.astype("int8")
        return X


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------

class EmailDomainFeatures(BaseEstimator, TransformerMixin):
    """Split P_emaildomain / R_emaildomain into provider + TLD, and add a
    boolean flag for whether purchaser and recipient share the same domain
    (high-signal feature on this dataset)."""

    PROVIDER_GROUPS = {
        "gmail":   {"gmail.com"},
        "yahoo":   {"yahoo.com", "yahoo.fr", "yahoo.es", "yahoo.de", "yahoo.co.uk", "ymail.com"},
        "outlook": {"outlook.com", "hotmail.com", "live.com", "msn.com"},
        "apple":   {"icloud.com", "me.com", "mac.com"},
        "aol":     {"aol.com"},
        "anonymous": {"anonymous.com"},
    }

    def fit(self, X, y=None):
        return self

    def _provider(self, value):
        if pd.isna(value):
            return "missing"
        for k, vs in self.PROVIDER_GROUPS.items():
            if value in vs:
                return k
        return "other"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in ("P_emaildomain", "R_emaildomain"):
            if col not in X.columns:
                continue
            X[f"{col}__provider"] = X[col].astype("object").apply(self._provider)
            X[f"{col}__tld"]      = (
                X[col].astype("object").str.rsplit(".", n=1).str[-1].fillna("missing")
            )
        if {"P_emaildomain", "R_emaildomain"}.issubset(X.columns):
            X["email_match"] = (
                X["P_emaildomain"].fillna("__a__") == X["R_emaildomain"].fillna("__b__")
            ).astype("int8")
        return X


# ---------------------------------------------------------------------------
# Frequency encoding (a classic Kaggle trick for high-cardinality columns)
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """For each given column, add a sibling column with the column's value
    frequency (computed on training data). Original column is preserved.

    Defaults target the high-signal high-cardinality columns on IEEE-CIS.
    """

    DEFAULT_COLS = ("card1", "card2", "card5", "addr1",
                    "P_emaildomain", "R_emaildomain")

    def __init__(self, columns: Optional[List[str]] = None,
                 suffix: str = "__freq"):
        self.columns = columns
        self.suffix  = suffix

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or [c for c in self.DEFAULT_COLS if c in X.columns]
        self.maps_ = {
            c: X[c].value_counts(dropna=False).to_dict() for c in cols
        }
        self.columns_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns_:
            if c not in X.columns:
                continue
            X[f"{c}{self.suffix}"] = (
                X[c].map(self.maps_[c]).fillna(0).astype("int32")
            )
        return X


# ---------------------------------------------------------------------------
# Group aggregations of TransactionAmt
# ---------------------------------------------------------------------------

class CardAggregations(BaseEstimator, TransformerMixin):
    """Per-card statistics of `TransactionAmt`. The implementation is
    leakage-safe: stats are computed on the training set in `fit` and
    *broadcast* at transform time via merge. Unseen group keys at
    inference yield NaN, which downstream imputation fills."""

    def __init__(
        self,
        group_specs: Optional[List[List[str]]] = None,
        target_col: str = "TransactionAmt",
    ):
        self.group_specs = group_specs
        self.target_col  = target_col

    def fit(self, X: pd.DataFrame, y=None):
        groups = self.group_specs or [["card1"], ["card1", "addr1"]]
        self.stats_ = {}
        for g in groups:
            if not all(c in X.columns for c in g + [self.target_col]):
                continue
            key = "_".join(g)
            grp = X.groupby(g, dropna=False)[self.target_col]
            self.stats_[key] = (
                g,
                grp.mean().rename(f"AMT_mean__{key}"),
                grp.std().rename(f"AMT_std__{key}"),
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.target_col not in X.columns:
            return X
        X = X.copy()
        for key, (g, mean_, std_) in self.stats_.items():
            X = X.merge(mean_.reset_index(), on=g, how="left")
            X = X.merge(std_.reset_index(),  on=g, how="left")
            mean_col = f"AMT_mean__{key}"
            X[f"AMT_ratio_to_mean__{key}"] = (
                X[self.target_col] / X[mean_col].replace(0, np.nan)
            )
        return X


# ---------------------------------------------------------------------------
# Top-level configurable FE — drives the *_FE_* MLflow runs
# ---------------------------------------------------------------------------

class IEEEFeatureEngineer(BaseEstimator, TransformerMixin):
    """Configurable feature-engineering pipeline.

    Each block is toggled by a boolean flag so a single class can drive
    many MLflow runs comparing FE strategies. Order matters: datetime &
    email features must come before frequency encoding so freq encoding
    can also see the new categorical columns if requested.
    """

    def __init__(
        self,
        use_datetime: bool = True,
        use_email:    bool = True,
        use_freq:     bool = True,
        use_agg:      bool = True,
        freq_columns: Optional[List[str]] = None,
        agg_groups:   Optional[List[List[str]]] = None,
    ):
        self.use_datetime = use_datetime
        self.use_email    = use_email
        self.use_freq     = use_freq
        self.use_agg      = use_agg
        self.freq_columns = freq_columns
        self.agg_groups   = agg_groups

    def _build_steps(self):
        steps = []
        if self.use_datetime:
            steps.append(("dt",   DatetimeFromTransactionDT()))
        if self.use_email:
            steps.append(("mail", EmailDomainFeatures()))
        if self.use_freq:
            steps.append(("freq", FrequencyEncoder(self.freq_columns)))
        if self.use_agg:
            steps.append(("agg",  CardAggregations(self.agg_groups)))
        return steps

    def fit(self, X: pd.DataFrame, y=None):
        self.steps_ = self._build_steps()
        Xc = X.copy()
        for _, s in self.steps_:
            s.fit(Xc, y)
            Xc = s.transform(Xc)
        # Remember the final column layout so transform() is deterministic
        self.feature_names_out_ = Xc.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        for _, s in self.steps_:
            Xc = s.transform(Xc)
        return Xc

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)
