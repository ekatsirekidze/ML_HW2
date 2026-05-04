"""MLflow + DagsHub helpers used by every notebook.

This module is the *only* place we touch MLflow setup. Notebooks import
the helpers below and never call `mlflow.set_tracking_uri` themselves —
that keeps configuration consistent and makes it trivial to swap remotes.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import cross_validate

import mlflow


# ---------------------------------------------------------------------------
# Tracking server setup
# ---------------------------------------------------------------------------

REQUIRED_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def setup_mlflow(experiment_name: Optional[str] = None) -> str:
    """Configure MLflow to point at the DagsHub remote.

    Reads credentials from environment variables — on Kaggle they should be
    populated from Kaggle Secrets earlier in the notebook:

        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        for k in ("MLFLOW_TRACKING_URI",
                  "MLFLOW_TRACKING_USERNAME",
                  "MLFLOW_TRACKING_PASSWORD"):
            os.environ[k] = s.get_secret(k)

    Returns
    -------
    str
        The active tracking URI (useful for logging into the notebook).
    """
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing MLflow env vars: {missing}. "
            "On Kaggle, attach the Secrets first."
        )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


# ---------------------------------------------------------------------------
# Run logging context manager
# ---------------------------------------------------------------------------

@contextmanager
def log_run(
    run_name: str,
    *,
    params: Optional[Dict] = None,
    tags: Optional[Dict] = None,
    nested: bool = False,
):
    """Open an MLflow run and stamp it with a consistent set of metadata.

    Usage::

        with log_run("XGBoost_Cleaning_v1_drop90",
                     params={"missing_threshold": 0.9},
                     tags={"stage": "cleaning", "model_family": "XGBoost"}):
            ...
            mlflow.log_metric("val_auc", 0.93)
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        yield run


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(
    model, X_train, y_train, X_val, y_val
) -> Dict[str, float]:
    """Fit on train, score on both train and val, return AUC + gap + timing.

    The `gap = train_auc - val_auc` metric is the single most important
    signal for "is this model overfitting" and should be logged on every run.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_sec = time.time() - t0
    p_train = model.predict_proba(X_train)[:, 1]
    p_val   = model.predict_proba(X_val)[:, 1]
    train_auc = float(roc_auc_score(y_train, p_train))
    val_auc   = float(roc_auc_score(y_val,   p_val))
    return {
        "train_auc": train_auc,
        "val_auc":   val_auc,
        "gap":       train_auc - val_auc,
        "fit_sec":   float(fit_sec),
    }


def cross_validate_auc(
    estimator, X, y, *, cv, n_jobs: int = 1
) -> Dict[str, float]:
    """Cross-validate AUC with both train and val scores per fold.

    Returns a dict suitable for passing to mlflow.log_metrics() plus the
    per-fold lists for plotting.
    """
    out = cross_validate(
        estimator, X, y,
        scoring="roc_auc",
        cv=cv,
        return_train_score=True,
        n_jobs=n_jobs,
    )
    return {
        "fold_train_auc":  list(map(float, out["train_score"])),
        "fold_val_auc":    list(map(float, out["test_score"])),
        "train_auc":       float(out["train_score"].mean()),
        "val_auc":         float(out["test_score"].mean()),
        "val_auc_std":     float(out["test_score"].std()),
        "gap":             float(out["train_score"].mean() - out["test_score"].mean()),
        "fit_sec":         float(out["fit_time"].mean()),
    }


# ---------------------------------------------------------------------------
# Plot helpers — all return matplotlib Figures so callers can do
# mlflow.log_figure(fig, "name.png")
# ---------------------------------------------------------------------------

def plot_roc(y_true, y_proba, *, title: str = "ROC curve") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, *, title: str = "Confusion @ 0.5") -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["legit", "fraud"])
    ax.set_yticklabels(["legit", "fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    names, importances, *, top: int = 30, title: str = "Feature importance"
) -> plt.Figure:
    names = np.asarray(names)
    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, max(4, top * 0.25)))
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: log a metrics dict atomically
# ---------------------------------------------------------------------------

def log_metrics_dict(metrics: Dict[str, float]) -> None:
    """Log every (k, v) pair where v is a float-coercible scalar.
    Lists (e.g. `fold_val_auc`) are logged as repeated step values."""
    for k, v in metrics.items():
        if isinstance(v, list):
            for step, val in enumerate(v):
                mlflow.log_metric(k, float(val), step=step)
        else:
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                mlflow.set_tag(k, str(v))
