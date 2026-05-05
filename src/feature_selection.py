"""Feature-selection helpers used during experimentation.

There are two patterns for using these:

1. **Compute offline, hard-code the list:**
   In a notebook, call `compute_mutual_info` / `compute_tree_importances` /
   `find_high_correlation` on a sample of training data to score features.
   Then plug the explicit list into the pipeline via `ColumnSubsetSelector`.
   This is the recommended approach because it makes the selection
   reproducible and visible in the notebook.

2. **In-pipeline:**
   Use sklearn's built-in `SelectKBest`, `SelectFromModel`, or
   `VarianceThreshold` after the `ColumnTransformer`. They work on the
   numeric matrix directly so no DataFrame plumbing is needed.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


# ---------------------------------------------------------------------------
# Offline analysis helpers
# ---------------------------------------------------------------------------

def find_high_correlation(
    df: pd.DataFrame, threshold: float = 0.95
) -> List[str]:
    """Return numeric columns to drop because they correlate > threshold
    with another numeric column. Keeps the first occurrence in column order."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if any(upper[c] > threshold)]


def compute_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample: Optional[int] = 100_000,
    random_state: int = 42,
) -> pd.Series:
    """Mutual information between each numeric column and the target.

    `mutual_info_classif` is O(n_samples * n_features * n_neighbors) so we
    sample by default. Pass `sample=None` to use the full data.
    """
    Xn = X.select_dtypes(include=[np.number]).fillna(-999)
    if sample is not None and len(Xn) > sample:
        idx = Xn.sample(sample, random_state=random_state).index
        Xn = Xn.loc[idx]
        ys = y.loc[idx]
    else:
        ys = y
    mi = mutual_info_classif(Xn, ys, random_state=random_state)
    return pd.Series(mi, index=Xn.columns).sort_values(ascending=False)


def compute_tree_importances(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Fit a tree-based estimator (RF, XGB, GBM …) and return its
    `feature_importances_` as a Series indexed by column name, sorted desc.

    Only numeric columns are used; NaNs are filled with -999 so the model
    can fit. Categorical columns must be encoded BEFORE calling this.
    """
    Xn = X.select_dtypes(include=[np.number]).fillna(-999)
    model.fit(Xn, y)
    return pd.Series(
        model.feature_importances_, index=Xn.columns
    ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# In-pipeline subset selector (stateless)
# ---------------------------------------------------------------------------

class ColumnSubsetSelector(BaseEstimator, TransformerMixin):
    """Keep only the named columns. Stateless — fit() is a no-op.

    Useful as the *last* step before the ColumnTransformer when the feature
    list has been chosen offline. Missing columns at inference time are
    silently skipped, so a model trained with feature `Xyz` doesn't crash
    if `Xyz` is missing from the raw test set.
    """

    def __init__(self, keep: List[str]):
        # Store as-is. sklearn.clone() does an identity check on constructor
        # params (`param1 is param2`); copying or wrapping (e.g. list(keep))
        # would break cross_validate / GridSearchCV with this transformer.
        self.keep = keep

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = self.keep if self.keep is not None else []
        present = [c for c in keep if c in X.columns]
        return X[present].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep)
