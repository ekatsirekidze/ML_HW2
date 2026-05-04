"""Reusable code shared by every model_experiment_*.ipynb notebook.

The `src/` package is intentionally small and focused:

  - data.py                  loaders for the four IEEE-CIS CSVs
  - preprocessing.py         sklearn-compatible cleaning transformers
  - feature_engineering.py   sklearn-compatible FE transformers
  - feature_selection.py     helpers + a ColumnSubsetSelector transformer
  - mlflow_utils.py          DagsHub/MLflow setup, log_run helper, plots

Every transformer subclasses BaseEstimator + TransformerMixin so the full
preprocessing chain can live inside a sklearn Pipeline and be pickled by
mlflow.sklearn.log_model().
"""

__version__ = "0.1.0"
