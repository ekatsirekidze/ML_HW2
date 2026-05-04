<!--
  ============================================================
  README skeleton — to be filled in during the project.
  Final version must be in Georgian (per assignment).
  An English working draft will live in README.en.md until the
  Georgian translation is complete.
  ============================================================
-->

# IEEE-CIS Fraud Detection — MLflow Experiment Tracking

> **Status:** work in progress (Phase 1 — repo + DagsHub connection)

This repository contains experiments for the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
Kaggle competition, with all runs tracked via MLflow on DagsHub.

The full README (in Georgian) will be written at the end of the project.
For the work-in-progress English version see [`README.en.md`](README.en.md).

## Quick links

- **Kaggle competition:** https://www.kaggle.com/competitions/ieee-fraud-detection
- **MLflow tracking server (DagsHub):** *to be added in Phase 1*
- **Best public LB score:** *to be added after submission*

## Repository layout (target)

```
.
├── README.md                  # this file (Georgian, final)
├── README.en.md               # English working draft
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── model_experiment_LogisticRegression.ipynb
│   ├── model_experiment_DecisionTree.ipynb
│   ├── model_experiment_Bagging.ipynb
│   ├── model_experiment_RandomForest.ipynb
│   ├── model_experiment_AdaBoost.ipynb
│   ├── model_experiment_GradientBoosting.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_MLP.ipynb
│   └── model_inference.ipynb
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── feature_selection.py
    └── mlflow_utils.py
```
