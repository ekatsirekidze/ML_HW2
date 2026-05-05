"""Generate the model_experiment_*.ipynb notebooks from a shared template.

The Logistic Regression notebook (`model_experiment_LogisticRegression.ipynb`)
was hand-written as the canonical template. Every other model notebook follows
the same six-section skeleton:

    1. Setup & MLflow Connection
    2. Cleaning              -> 3 MLflow runs
    3. Feature Engineering   -> 5 MLflow runs (cumulative)
    4. Feature Selection     -> 4 MLflow runs (all / corr / MI / model-based)
    5. Training              -> 6+ MLflow runs (incl. underfit + overfit demos)
                                + a parameter sweep with nested child runs
    6. Final Pipeline        -> StratifiedKFold vs TimeSeriesSplit comparison
                                + final fit on full data + Model Registry

This script keeps all 7 model notebooks consistent with that skeleton;
per-model differences (model class, pipeline shape, HPO grid) live in the
`MODELS` config dict below.

Run from the repo root:

    python tools/build_model_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


# =============================================================================
# Per-model configuration
# =============================================================================
#
# Each model defines:
#
# `import_block`   : `from X import Y` line(s) that bring in the model class
# `model_family`   : tag value, also used in the experiment + run names
# `factory_body`   : the body of `make_<m>_pipeline(...)` -- everything after
#                    the cleaner+FE steps.  Different model families need
#                    different preprocessing (e.g. trees skip scaling, XGB
#                    handles NaN natively).
# `default_kwargs` : kwargs for the baseline run.
# `train_runs`     : list of (run_name, kwargs, purpose|None).
# `sweep`          : (param_name, list_of_values, fixed_kwargs) for the
#                    nested-children HPO sweep.
# `final_kwargs`   : kwargs for the final, full-data fit + Model Registry.
# `extra_install`  : optional pip-install line for the model package
#                    (e.g. xgboost on Kaggle is preinstalled, but we add the
#                    install line defensively).
# =============================================================================

MODELS = {
    # -------------------------------------------------------------------------
    "DecisionTree": dict(
        import_block="from sklearn.tree import DecisionTreeClassifier",
        model_family="DecisionTree",
        model_class="DecisionTreeClassifier",
        # Trees: no scaling needed; impute numerics with -999, ordinal-encode cats.
        preprocessor="""
        from sklearn.preprocessing import OrdinalEncoder
        num_pipe = SimpleImputer(strategy="constant", fill_value=-999.0)
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(max_depth=8, min_samples_leaf=20,
                            random_state="RANDOM_STATE", class_weight="None"),
        train_runs=[
            ("train_v1_baseline_depth8",   dict(max_depth=8,    min_samples_leaf=20),                          None),
            ("train_v2_depth2_underfit",   dict(max_depth=2,    min_samples_leaf=20),                          "underfit_demo"),
            ("train_v3_depthNone_overfit", dict(max_depth="None", min_samples_leaf=1),                         "overfit_demo"),
            ("train_v4_balanced",          dict(max_depth=8,    min_samples_leaf=20, class_weight='"balanced"'), None),
            ("train_v5_min_leaf_5",        dict(max_depth=10,   min_samples_leaf=5),                           None),
            ("train_v6_entropy",           dict(max_depth=8,    min_samples_leaf=20, criterion='"entropy"'),   None),
        ],
        sweep=("max_depth", [3, 5, 8, 12, 20, "None"], dict(min_samples_leaf=20)),
        final_kwargs=dict(max_depth=8, min_samples_leaf=20),
        extra_install=None,
    ),

    # -------------------------------------------------------------------------
    "Bagging": dict(
        import_block="from sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier",
        model_family="Bagging",
        model_class="BaggingClassifier",
        preprocessor="""
        from sklearn.preprocessing import OrdinalEncoder
        num_pipe = SimpleImputer(strategy="constant", fill_value=-999.0)
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(
            estimator='DecisionTreeClassifier(max_depth=8, min_samples_leaf=20)',
            n_estimators=50, max_samples=0.7, max_features=1.0,
            n_jobs=-1, random_state="RANDOM_STATE",
        ),
        train_runs=[
            ("train_v1_baseline_50est",      dict(n_estimators=50, max_samples=0.7),                           None),
            ("train_v2_10est_underfit",      dict(n_estimators=10, max_samples=0.5),                           "underfit_demo"),
            ("train_v3_300est_deepbase",     dict(n_estimators=300, max_samples=1.0,
                                                  estimator='DecisionTreeClassifier(max_depth=None)'),         "overfit_demo"),
            ("train_v4_subsample_0_5",       dict(n_estimators=100, max_samples=0.5),                          None),
            ("train_v5_max_features_0_5",    dict(n_estimators=100, max_samples=0.7, max_features=0.5),        None),
            ("train_v6_shallow_base",        dict(n_estimators=100, max_samples=0.7,
                                                  estimator='DecisionTreeClassifier(max_depth=4)'),            None),
        ],
        sweep=("n_estimators", [10, 50, 100, 200], dict(max_samples=0.7, max_features=0.7)),
        final_kwargs=dict(n_estimators=200, max_samples=0.7, max_features=0.7),
        extra_install=None,
    ),

    # -------------------------------------------------------------------------
    "RandomForest": dict(
        import_block="from sklearn.ensemble import RandomForestClassifier",
        model_family="RandomForest",
        model_class="RandomForestClassifier",
        preprocessor="""
        from sklearn.preprocessing import OrdinalEncoder
        num_pipe = SimpleImputer(strategy="constant", fill_value=-999.0)
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(
            n_estimators=200, max_depth=12, max_features='"sqrt"',
            min_samples_leaf=5, n_jobs=-1, random_state="RANDOM_STATE",
        ),
        train_runs=[
            ("train_v1_baseline_200_d12",        dict(n_estimators=200, max_depth=12),                                None),
            ("train_v2_50_d3_underfit",          dict(n_estimators=50,  max_depth=3),                                 "underfit_demo"),
            ("train_v3_500_dNone_overfit",       dict(n_estimators=500, max_depth="None", min_samples_leaf=1),        "overfit_demo"),
            ("train_v4_balanced",                dict(n_estimators=200, max_depth=12, class_weight='"balanced"'),     None),
            ("train_v5_max_features_log2",       dict(n_estimators=200, max_depth=12, max_features='"log2"'),         None),
            ("train_v6_min_leaf_50",             dict(n_estimators=200, max_depth=12, min_samples_leaf=50),           None),
        ],
        sweep=("max_depth", [4, 8, 12, 16, "None"], dict(n_estimators=200)),
        final_kwargs=dict(n_estimators=400, max_depth=14, min_samples_leaf=5, max_features='"sqrt"'),
        extra_install=None,
    ),

    # -------------------------------------------------------------------------
    "AdaBoost": dict(
        import_block="from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier",
        model_family="AdaBoost",
        model_class="AdaBoostClassifier",
        preprocessor="""
        from sklearn.preprocessing import OrdinalEncoder
        num_pipe = SimpleImputer(strategy="constant", fill_value=-999.0)
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(
            n_estimators=100, learning_rate=1.0,
            estimator='DecisionTreeClassifier(max_depth=3)',
            random_state="RANDOM_STATE",
        ),
        train_runs=[
            ("train_v1_baseline_100_lr1",     dict(n_estimators=100, learning_rate=1.0),                                  None),
            ("train_v2_10_lr0_5_underfit",    dict(n_estimators=10,  learning_rate=0.5),                                  "underfit_demo"),
            ("train_v3_lr2_diverge",          dict(n_estimators=200, learning_rate=2.0),                                  "overfit_demo"),
            ("train_v4_deeperbase",           dict(n_estimators=100, learning_rate=1.0,
                                                    estimator='DecisionTreeClassifier(max_depth=6)'),                    None),
            ("train_v5_lr0_1",                dict(n_estimators=200, learning_rate=0.1),                                  None),
            ("train_v6_500_lr0_5",            dict(n_estimators=500, learning_rate=0.5),                                  None),
        ],
        sweep=("learning_rate", [0.05, 0.1, 0.5, 1.0, 1.5], dict(n_estimators=100)),
        final_kwargs=dict(n_estimators=300, learning_rate=0.5),
        extra_install=None,
    ),

    # -------------------------------------------------------------------------
    "GradientBoosting": dict(
        import_block="from sklearn.experimental import enable_hist_gradient_boosting  # noqa\nfrom sklearn.ensemble import HistGradientBoostingClassifier as GradientBoostingClassifier",
        model_family="GradientBoosting",
        model_class="GradientBoostingClassifier",
        # HistGradientBoostingClassifier handles NaN natively. No imputation needed.
        # Categorical columns must be ordinal-encoded.
        preprocessor="""
        from sklearn.preprocessing import OrdinalEncoder
        num_pipe = "passthrough"
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(
            max_iter=200, learning_rate=0.1, max_depth=8,
            min_samples_leaf=20, random_state="RANDOM_STATE",
        ),
        train_runs=[
            ("train_v1_baseline_200_lr0_1",     dict(max_iter=200, learning_rate=0.1, max_depth=8),                None),
            ("train_v2_50_d2_underfit",         dict(max_iter=50,  learning_rate=0.05, max_depth=2),               "underfit_demo"),
            ("train_v3_500_lr0_3_overfit",      dict(max_iter=500, learning_rate=0.3,  max_depth=12,
                                                      min_samples_leaf=1),                                          "overfit_demo"),
            ("train_v4_lr0_05_iter300",         dict(max_iter=300, learning_rate=0.05, max_depth=8),               None),
            ("train_v5_l2_reg",                 dict(max_iter=200, learning_rate=0.1, max_depth=8, l2_regularization=1.0), None),
            ("train_v6_balanced",               dict(max_iter=200, learning_rate=0.1, max_depth=8, class_weight='"balanced"'), None),
        ],
        sweep=("learning_rate", [0.01, 0.05, 0.1, 0.2], dict(max_iter=200, max_depth=8)),
        final_kwargs=dict(max_iter=400, learning_rate=0.05, max_depth=8, l2_regularization=0.5),
        extra_install=None,
    ),

    # -------------------------------------------------------------------------
    "XGBoost": dict(
        import_block="from xgboost import XGBClassifier",
        model_family="XGBoost",
        model_class="XGBClassifier",
        # XGBoost handles NaN natively for numerics; for categoricals we ordinal-encode.
        preprocessor="""
        from sklearn.preprocessing import OrdinalEncoder
        num_pipe = "passthrough"
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord",    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            tree_method='"hist"', eval_metric='"auc"',
            random_state="RANDOM_STATE", n_jobs=-1, verbosity=0,
        ),
        train_runs=[
            ("train_v1_baseline_400",          dict(n_estimators=400, max_depth=6, learning_rate=0.1),                None),
            ("train_v2_d2_50_underfit",        dict(n_estimators=50,  max_depth=2, learning_rate=0.05),               "underfit_demo"),
            ("train_v3_d15_lr0_3_overfit",     dict(n_estimators=600, max_depth=15, learning_rate=0.3,
                                                      reg_alpha=0, reg_lambda=0, subsample=1.0),                       "overfit_demo"),
            ("train_v4_lr0_03_long",           dict(n_estimators=800, max_depth=6, learning_rate=0.03),               None),
            ("train_v5_high_reg",              dict(n_estimators=400, max_depth=6, learning_rate=0.1,
                                                      reg_alpha=1.0, reg_lambda=1.0),                                  None),
            ("train_v6_scale_pos_weight",      dict(n_estimators=400, max_depth=6, learning_rate=0.1,
                                                      scale_pos_weight=27),                                            None),
        ],
        sweep=("max_depth", [3, 5, 6, 8, 10], dict(n_estimators=400, learning_rate=0.1)),
        final_kwargs=dict(n_estimators=800, max_depth=6, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0),
        extra_install="xgboost",
    ),

    # -------------------------------------------------------------------------
    "MLP": dict(
        import_block="from sklearn.neural_network import MLPClassifier",
        model_family="MLP",
        model_class="MLPClassifier",
        # MLP needs scaled, dense, NaN-free input.
        preprocessor="""
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=np.number)),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ],
            remainder="drop", verbose_feature_names_out=False,
        )
        """.strip(),
        default_kwargs=dict(
            hidden_layer_sizes='(128, 64)', activation='"relu"', solver='"adam"',
            alpha=1e-4, batch_size=512, learning_rate_init=1e-3, max_iter=30,
            early_stopping=True, n_iter_no_change=5, random_state="RANDOM_STATE",
        ),
        train_runs=[
            ("train_v1_baseline_128_64",        dict(hidden_layer_sizes='(128, 64)'),                                                                None),
            ("train_v2_8_underfit",             dict(hidden_layer_sizes='(8,)', max_iter=20),                                                        "underfit_demo"),
            ("train_v3_512_256_128_overfit",    dict(hidden_layer_sizes='(512, 256, 128)', alpha=0.0, max_iter=80, early_stopping=False),            "overfit_demo"),
            ("train_v4_high_alpha",             dict(hidden_layer_sizes='(128, 64)', alpha=1e-2),                                                    None),
            ("train_v5_lower_lr",               dict(hidden_layer_sizes='(128, 64)', learning_rate_init=1e-4, max_iter=60),                          None),
            ("train_v6_tanh",                   dict(hidden_layer_sizes='(128, 64)', activation='"tanh"'),                                           None),
        ],
        sweep=("hidden_layer_sizes",
               ['(64,)', '(128,)', '(128, 64)', '(256, 128)', '(256, 128, 64)'],
               dict(max_iter=30, early_stopping=True)),
        final_kwargs=dict(hidden_layer_sizes='(256, 128, 64)', alpha=1e-4,
                          learning_rate_init=1e-3, max_iter=60, early_stopping=True),
        extra_install=None,
    ),
}


# =============================================================================
# Cell builders
# =============================================================================

def _kw_repr(kwargs: dict) -> str:
    """Render a kwargs dict as ``a=1, b="x"`` Python source.

    Values that are already Python source (e.g. "RANDOM_STATE", "(64, 32)",
    `'"sqrt"'`) are inlined verbatim; numeric / bool / None values are
    `repr()`-ed normally.
    """
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip().split("\n")],
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in dedent(text).strip().split("\n")],
    }


def _code_raw(lines: list[str]) -> dict:
    """Build a code cell from already-properly-indented lines (no dedent).

    Use this for cells that interpolate multi-line strings, since dedent
    can't reason about pre-formatted blocks injected via f-strings.
    """
    src = "\n".join(lines).rstrip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.split("\n")],
    }


def _indent(text: str, spaces: int) -> str:
    """Indent every non-empty line of `text` by `spaces` spaces."""
    pad = " " * spaces
    return "\n".join((pad + ln) if ln.strip() else "" for ln in text.split("\n"))


def _normalise_block(text: str) -> str:
    """Bring an arbitrarily-indented multi-line string to column 0.

    The preprocessor blocks in `MODELS` were defined as triple-quoted strings
    surrounded by 8 spaces (the dict-literal indent level), then `.strip()`-ed
    to remove the wrapping blank lines. The .strip() also removes the leading
    indent of the very first line, which makes textwrap.dedent give up. This
    helper recovers the intended column-0 layout by:

      * finding the minimum indent across the *non-blank, non-first* lines,
      * stripping that much from every non-blank line, AND
      * leaving any line that already has 0 indent untouched.
    """
    lines = text.expandtabs(4).split("\n")
    non_blank = [ln for ln in lines if ln.strip()]
    if len(non_blank) <= 1:
        return text

    # First line may have been .strip()-damaged; use the others as reference.
    rest_indents = [len(ln) - len(ln.lstrip()) for ln in non_blank[1:]]
    common = min(rest_indents) if rest_indents else 0
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        elif ln[:common].isspace() or ln[:common] == "":
            # Has at least `common` spaces of indent OR is at column 0 already
            out.append(ln[common:] if (len(ln) - len(ln.lstrip())) >= common else ln)
        else:
            out.append(ln)
    return "\n".join(out).strip("\n")


def build_notebook(name: str, cfg: dict) -> dict:
    family = cfg["model_family"]
    cls    = cfg["model_class"]
    install_line = (
        f'    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "{cfg["extra_install"]}"])'
        if cfg.get("extra_install") else ""
    )
    install_block = (
        f"    # Ensure {cfg['extra_install']} is installed (preinstalled on Kaggle, install on local)\n"
        f"    try:\n        __import__(\"{cfg['extra_install']}\")\n    except ImportError:\n{install_line}\n"
    ) if cfg.get("extra_install") else ""

    cells = []

    # -------- 0. Title --------
    cells.append(_md(f"""
        # {family} - Model Experiments

        This notebook runs all {family} experiments and logs them to the MLflow
        experiment **`{family}_Training`** on DagsHub.

        ## Notebook structure (required by the assignment)

        1. Setup & MLflow Connection
        2. Cleaning
        3. Feature Engineering
        4. Feature Selection
        5. Training (incl. hyperparameter sweep + over-/under-fitting demos)
        6. Final Pipeline & Logging
    """))

    # -------- 1. Setup --------
    cells.append(_md("# 1. Setup & MLflow Connection"))

    setup_lines = [
        "# =============================================================",
        "# Setup - auto-detects environment and locates the `src/` package.",
        "# See model_experiment_LogisticRegression.ipynb for the full notes.",
        "# =============================================================",
        "import os, sys, subprocess, shutil",
        "",
        'REPO_URL = "https://github.com/<YOUR_GH_USERNAME>/<YOUR_REPO_NAME>.git"   # only used for git-clone fallback',
        "",
        'ON_KAGGLE = os.path.exists("/kaggle/input")',
        "SRC_FOUND = None",
        "",
        "if ON_KAGGLE:",
        '    for _d in os.listdir("/kaggle/input"):',
        '        _candidate = f"/kaggle/input/{_d}"',
        '        if os.path.isdir(os.path.join(_candidate, "src")):',
        "            SRC_FOUND = _candidate",
        "            break",
        "    if SRC_FOUND is None:",
        '        REPO_DIR = "/kaggle/working/repo"',
        "        if os.path.isdir(REPO_DIR):",
        "            shutil.rmtree(REPO_DIR)",
        '        subprocess.check_call(["git", "clone", "-q", REPO_URL, REPO_DIR])',
        "        SRC_FOUND = REPO_DIR",
        "    sys.path.insert(0, SRC_FOUND)",
        "",
        '    for _pkg in ("mlflow==2.10.2", "dagshub"):',
        "        try:",
        '            __import__(_pkg.split("==")[0].replace("-", "_"))',
        "        except ImportError:",
        '            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])',
    ]
    if cfg.get("extra_install"):
        pkg = cfg["extra_install"]
        setup_lines += [
            f"    # Ensure {pkg} is installed (pre-installed on Kaggle but defensive)",
            "    try:",
            f'        __import__("{pkg}")',
            "    except ImportError:",
            f'        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "{pkg}"])',
        ]
    setup_lines += [
        "",
        "    from kaggle_secrets import UserSecretsClient",
        "    _s = UserSecretsClient()",
        '    for _k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):',
        "        os.environ[_k] = _s.get_secret(_k)",
        "",
        '    print("src/ found at :", SRC_FOUND)',
        "else:",
        '    sys.path.insert(0, "..")',
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
    ]
    # import_block may contain multiple imports separated by \n
    setup_lines += cfg["import_block"].split("\n")
    setup_lines += [
        "from sklearn.pipeline import Pipeline",
        "from sklearn.compose import ColumnTransformer, make_column_selector",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit",
        "",
        "from src.data import load_train, downcast_numerics",
        "from src.preprocessing import IEEECleaner",
        "from src.feature_engineering import IEEEFeatureEngineer",
        "from src.feature_selection import (",
        "    find_high_correlation, compute_mutual_info, ColumnSubsetSelector,",
        ")",
        "from src.mlflow_utils import (",
        "    setup_mlflow, log_run, log_metrics_dict,",
        "    evaluate_holdout, cross_validate_auc,",
        "    plot_roc, plot_confusion, plot_feature_importance,",
        ")",
        "import mlflow, mlflow.sklearn",
        "",
        f'setup_mlflow("{family}_Training")',
        '''print("Environment :", "Kaggle" if ON_KAGGLE else "Local")''',
        '''print("Tracking URI:", mlflow.get_tracking_uri())''',
    ]
    cells.append(_code_raw(setup_lines))

    # ---- data load + holdout ----
    cells.append(_md("""
        ## 1.1 Load data and create a time-based holdout

        Same setup as the LR notebook: sort by `TransactionDT`, hold out the
        last 20% in time.  Most exploratory runs use a 100k stratified subsample;
        the *final* model retrains on full data.
    """))

    cells.append(_code("""
        RANDOM_STATE  = 42
        SUBSAMPLE     = 100_000
        HOLDOUT_FRAC  = 0.20

        df = downcast_numerics(load_train())
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        split_idx = int(len(df) * (1 - HOLDOUT_FRAC))
        train_full = df.iloc[:split_idx]
        val_full   = df.iloc[split_idx:]

        def subsample_stratified(d, n, seed=RANDOM_STATE):
            if len(d) <= n:
                return d
            pos = d[d["isFraud"] == 1]
            neg_pool = d[d["isFraud"] == 0]
            need = n - len(pos)
            neg = neg_pool.sample(need, random_state=seed) if need < len(neg_pool) else neg_pool
            return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

        train_small = subsample_stratified(train_full, SUBSAMPLE)

        X_train_s, y_train_s = train_small.drop(columns=["isFraud"]), train_small["isFraud"]
        X_val,     y_val     = val_full.drop(columns=["isFraud"]),    val_full["isFraud"]
        X_train_f, y_train_f = train_full.drop(columns=["isFraud"]),  train_full["isFraud"]

        print(f"train (full)  : {X_train_f.shape}  | fraud rate {y_train_f.mean():.3%}")
        print(f"train (small) : {X_train_s.shape}  | fraud rate {y_train_s.mean():.3%}")
        print(f"val           : {X_val.shape}  | fraud rate {y_val.mean():.3%}")
    """))

    # ---- pipeline factory ----
    cells.append(_md(f"""
        ## 1.2 Pipeline factory

        Same factory pattern as the LR notebook.  Differences for {family}:

        {_factory_notes(family)}
    """))

    factory_default = _kw_repr(cfg["default_kwargs"])
    factory_lines = [
        f"def make_{family.lower()}_pipeline(",
        "    *,",
        "    cleaner_kwargs: dict | None = None,",
        "    fe_kwargs:      dict | None = None,",
        "    model_kwargs:   dict | None = None,",
        ") -> Pipeline:",
        "    cleaner_kwargs = cleaner_kwargs or {}",
        "    fe_kwargs      = fe_kwargs      or {}",
        f"    base_kwargs    = dict({factory_default})",
        "    base_kwargs.update(model_kwargs or {})",
        "",
    ]
    # Dedent the preprocessor block (it has residual indent from the config
    # literal), then re-indent every line by 4 spaces so it sits inside the
    # function body.
    pp_dedented = _normalise_block(cfg["preprocessor"])
    factory_lines += [
        ("    " + ln) if ln.strip() else "" for ln in pp_dedented.split("\n")
    ]
    factory_lines += [
        "",
        "    return Pipeline([",
        '        ("clean", IEEECleaner(**cleaner_kwargs)),',
        '        ("fe",    IEEEFeatureEngineer(**fe_kwargs)),',
        '        ("prep",  preprocessor),',
        f'        ("model", {cls}(**base_kwargs)),',
        "    ])",
        "",
        "# Smoke-test",
        f"print(make_{family.lower()}_pipeline())",
    ]
    cells.append(_code_raw(factory_lines))

    # -------- 2. Cleaning --------
    cells.append(_md("""
        # 2. Cleaning

        Three cleaning strategies (same as LR notebook so results compare cleanly).
    """))

    cells.append(_code(f"""
        cleaning_grid = [
            dict(run="cleaning_v1_drop95",  cleaner=dict(missing_threshold=0.95)),
            dict(run="cleaning_v2_drop90",  cleaner=dict(missing_threshold=0.90)),
            dict(run="cleaning_v3_drop99",  cleaner=dict(missing_threshold=0.99)),
        ]

        cleaning_results = {{}}
        for cfg_ in cleaning_grid:
            with log_run(
                cfg_["run"],
                params={{**cfg_["cleaner"], "subsample": SUBSAMPLE, "fe_active": False}},
                tags={{"model_family": "{family}", "stage": "cleaning"}},
            ):
                pipe = make_{family.lower()}_pipeline(
                    cleaner_kwargs=cfg_["cleaner"],
                    fe_kwargs=dict(use_datetime=False, use_email=False,
                                    use_freq=False, use_agg=False),
                )
                m = evaluate_holdout(pipe, X_train_s, y_train_s, X_val, y_val)
                log_metrics_dict(m)
                cleaning_results[cfg_["run"]] = m
                print(f"{{cfg_['run']:25}} train={{m['train_auc']:.4f}}  val={{m['val_auc']:.4f}}  gap={{m['gap']:+.4f}}")

        best_cleaning      = max(cleaning_results, key=lambda k: cleaning_results[k]["val_auc"])
        BEST_CLEANING_KW   = next(c for c in cleaning_grid if c["run"] == best_cleaning)["cleaner"]
        print(f"\\n>> best cleaning: {{best_cleaning}}")
    """))

    # -------- 3. Feature Engineering --------
    cells.append(_md("""
        # 3. Feature Engineering

        Cumulative FE blocks (same five rungs as in the LR notebook).
    """))

    cells.append(_code(f"""
        fe_grid = [
            ("fe_v1_baseline",  dict(use_datetime=False, use_email=False, use_freq=False, use_agg=False)),
            ("fe_v2_+datetime", dict(use_datetime=True,  use_email=False, use_freq=False, use_agg=False)),
            ("fe_v3_+email",    dict(use_datetime=True,  use_email=True,  use_freq=False, use_agg=False)),
            ("fe_v4_+freq",     dict(use_datetime=True,  use_email=True,  use_freq=True,  use_agg=False)),
            ("fe_v5_full",      dict(use_datetime=True,  use_email=True,  use_freq=True,  use_agg=True)),
        ]

        fe_results = {{}}
        for run_name, fe_kw in fe_grid:
            with log_run(
                run_name,
                params={{**fe_kw, "subsample": SUBSAMPLE, "cleaning": best_cleaning}},
                tags={{"model_family": "{family}", "stage": "fe"}},
            ):
                pipe = make_{family.lower()}_pipeline(
                    cleaner_kwargs=BEST_CLEANING_KW, fe_kwargs=fe_kw,
                )
                m = evaluate_holdout(pipe, X_train_s, y_train_s, X_val, y_val)
                log_metrics_dict(m)
                fe_results[run_name] = m
                print(f"{{run_name:20}} train={{m['train_auc']:.4f}}  val={{m['val_auc']:.4f}}  gap={{m['gap']:+.4f}}")

        best_fe_name = max(fe_results, key=lambda k: fe_results[k]["val_auc"])
        BEST_FE_KW   = dict(fe_grid)[best_fe_name]
        print(f"\\n>> best FE: {{best_fe_name}}")
    """))

    # -------- 4. Feature Selection --------
    cells.append(_md("""
        # 4. Feature Selection

        Four strategies, each operating on the source-column level.
    """))

    cells.append(_code("""
        # Materialise the cleaned + FE'd training subsample for selection analysis.
        prep_view = Pipeline([
            ("clean", IEEECleaner(**BEST_CLEANING_KW)),
            ("fe",    IEEEFeatureEngineer(**BEST_FE_KW)),
        ]).fit(X_train_s, y_train_s).transform(X_train_s)

        cat_cols_src = prep_view.select_dtypes(include="object").columns.tolist()
        num_cols_src = prep_view.select_dtypes(include=np.number).columns.tolist()

        all_cols = prep_view.columns.tolist()

        high_corr = find_high_correlation(prep_view[num_cols_src], threshold=0.95)
        fs_v2_keep = [c for c in all_cols if c not in high_corr]

        mi = compute_mutual_info(prep_view[num_cols_src], y_train_s, sample=50_000)
        fs_v3_keep = list(dict.fromkeys(mi.head(200).index.tolist() + cat_cols_src))

        # Tree-importance based selection
        from sklearn.ensemble import RandomForestClassifier
        rf_probe = RandomForestClassifier(
            n_estimators=80, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE,
        )
        rf_probe.fit(prep_view[num_cols_src].fillna(-999), y_train_s)
        rf_imp = pd.Series(rf_probe.feature_importances_, index=num_cols_src)
        fs_v4_keep = list(dict.fromkeys(
            rf_imp.sort_values(ascending=False).head(200).index.tolist() + cat_cols_src
        ))

        print(f"v1_all          : {len(all_cols):4d}")
        print(f"v2_drop_corr95  : {len(fs_v2_keep):4d}")
        print(f"v3_top_mi200    : {len(fs_v3_keep):4d}")
        print(f"v4_top_rfimp200 : {len(fs_v4_keep):4d}")
    """))

    cells.append(_code(f"""
        def make_pipeline_with_subset(keep_cols, **kw):
            base = make_{family.lower()}_pipeline(**kw)
            if keep_cols is None:
                return base
            steps = list(base.steps)
            fe_idx = next(i for i, (n, _) in enumerate(steps) if n == "fe")
            steps.insert(fe_idx + 1, ("subset", ColumnSubsetSelector(keep_cols)))
            return Pipeline(steps)

        fs_grid = [
            ("fs_v1_all",           None),
            ("fs_v2_drop_corr95",   fs_v2_keep),
            ("fs_v3_top_mi200",     fs_v3_keep),
            ("fs_v4_top_rfimp200",  fs_v4_keep),
        ]

        fs_results = {{}}
        for run_name, keep in fs_grid:
            with log_run(
                run_name,
                params={{"n_features_kept": (len(keep) if keep else len(all_cols)),
                        "subsample": SUBSAMPLE,
                        "cleaning":  best_cleaning,
                        "fe":        best_fe_name}},
                tags={{"model_family": "{family}", "stage": "fs"}},
            ):
                pipe = make_pipeline_with_subset(
                    keep, cleaner_kwargs=BEST_CLEANING_KW, fe_kwargs=BEST_FE_KW,
                )
                m = evaluate_holdout(pipe, X_train_s, y_train_s, X_val, y_val)
                log_metrics_dict(m)
                fs_results[run_name] = m
                print(f"{{run_name:22}} train={{m['train_auc']:.4f}}  val={{m['val_auc']:.4f}}  gap={{m['gap']:+.4f}}")

        best_fs_name = max(fs_results, key=lambda k: fs_results[k]["val_auc"])
        BEST_FS_KEEP = dict(fs_grid)[best_fs_name]
        print(f"\\n>> best FS: {{best_fs_name}}")
    """))

    # -------- 5. Training --------
    cells.append(_md(f"""
        # 5. Training

        Now cleaning + FE + FS are locked in.  We sweep {family} hyperparameters
        and explicitly include over-/under-fit demos.
    """))

    train_lines = []
    for run_name, kw, purpose in cfg["train_runs"]:
        purpose_arg = f', purpose="{purpose}"' if purpose else ""
        kw_src = _kw_repr(kw)
        train_lines.append(
            f"        run_lr_training(\"{run_name}\", dict({kw_src}){purpose_arg})"
        )
    train_block = "\n".join(train_lines)

    cells.append(_code(f"""
        # --- Precompute the heavy cleaning + FE + FS + preprocessing once -------
        # Section 5 only varies the {family} hyperparameters; cleaning / FE / FS
        # / preprocessing are identical for every run, so re-running them inside
        # every pipeline wastes minutes per run.  We build the pipeline once,
        # drop the final model step, fit on the training subsample, transform
        # both train + holdout to matrices, and then per-experiment fit only
        # the {cls}(...) step on the precomputed matrices.  The MLflow logs
        # (params, metrics, plots) are byte-for-byte identical to the
        # pipeline-based version.
        import time
        from sklearn.metrics import roc_auc_score

        print("Precomputing transformed matrices for the {family} training sweep ...")
        _t0 = time.time()
        _prep_only = make_pipeline_with_subset(
            BEST_FS_KEEP,
            cleaner_kwargs=BEST_CLEANING_KW,
            fe_kwargs=BEST_FE_KW,
            model_kwargs=None,
        )
        _prep_only.steps = _prep_only.steps[:-1]                  # drop the model step
        _prep_only.fit(X_train_s, y_train_s)
        X_tr_mat  = _prep_only.transform(X_train_s)
        X_val_mat = _prep_only.transform(X_val)
        print(f"  done in {{time.time()-_t0:.1f}}s | train shape={{X_tr_mat.shape}}")

        MODEL_DEFAULT_KW = dict({factory_default})


        def run_lr_training(run_name, model_kwargs, *, purpose=None, extra_params=None):
            base_kwargs = dict(MODEL_DEFAULT_KW)
            base_kwargs.update(model_kwargs or {{}})
            model = {cls}(**base_kwargs)

            with log_run(
                run_name,
                params={{
                    **{{k: str(v) for k, v in model_kwargs.items()}},
                    "subsample":      SUBSAMPLE,
                    "cleaning":       best_cleaning,
                    "fe":             best_fe_name,
                    "feature_select": best_fs_name,
                    **(extra_params or {{}}),
                }},
                tags={{"model_family": "{family}", "stage": "training",
                        **({{"purpose": purpose}} if purpose else {{}})}},
            ):
                t0 = time.time()
                model.fit(X_tr_mat, y_train_s)
                fit_sec = time.time() - t0
                p_tr  = model.predict_proba(X_tr_mat)[:, 1]
                p_val = model.predict_proba(X_val_mat)[:, 1]
                tr_auc  = float(roc_auc_score(y_train_s, p_tr))
                val_auc = float(roc_auc_score(y_val,     p_val))
                m = {{"train_auc": tr_auc, "val_auc": val_auc,
                      "gap": tr_auc - val_auc, "fit_sec": float(fit_sec)}}
                log_metrics_dict(m)
                mlflow.log_figure(plot_roc(y_val, p_val, title=run_name), "roc.png")
                mlflow.log_figure(plot_confusion(y_val, (p_val >= 0.5).astype(int),
                                                  title=run_name), "confusion.png")
                print(f"{{run_name:34}} train={{tr_auc:.4f}}  val={{val_auc:.4f}}  "
                      f"gap={{tr_auc-val_auc:+.4f}}  ({{fit_sec:.1f}}s)")
                return m

{train_block}
    """))

    # ---- HPO sweep ----
    sweep_param, sweep_vals, sweep_fixed = cfg["sweep"]
    sweep_vals_src = "[" + ", ".join(str(v) for v in sweep_vals) + "]"
    sweep_fixed_src = _kw_repr(sweep_fixed)
    cells.append(_code(f"""
        # --- HPO sweep: reuses the X_tr_mat / X_val_mat from the cell above ---
        sweep_grid = {sweep_vals_src}
        sweep_rows = []

        with log_run(
            "train_v7_{sweep_param}_sweep_parent",
            params={{"sweep_param": "{sweep_param}", "sweep_grid": str(sweep_grid)}},
            tags={{"model_family": "{family}", "stage": "hpo"}},
        ):
            for v in sweep_grid:
                with log_run(
                    f"train_v7_{sweep_param}={{v}}",
                    params={{"{sweep_param}": v, **{{k: str(vv) for k, vv in dict({sweep_fixed_src}).items()}}}},
                    tags={{"model_family": "{family}", "stage": "hpo", "parent": "{sweep_param}_sweep"}},
                    nested=True,
                ):
                    base_kwargs = dict(MODEL_DEFAULT_KW)
                    base_kwargs.update(dict({sweep_param}=v, {sweep_fixed_src}))
                    model = {cls}(**base_kwargs)
                    t0 = time.time()
                    model.fit(X_tr_mat, y_train_s)
                    fit_sec = time.time() - t0
                    p_tr  = model.predict_proba(X_tr_mat)[:, 1]
                    p_val = model.predict_proba(X_val_mat)[:, 1]
                    tr_auc  = float(roc_auc_score(y_train_s, p_tr))
                    val_auc = float(roc_auc_score(y_val,     p_val))
                    m = {{"train_auc": tr_auc, "val_auc": val_auc,
                          "gap": tr_auc - val_auc, "fit_sec": float(fit_sec)}}
                    log_metrics_dict(m)
                    sweep_rows.append({{"{sweep_param}": v, **m}})
                    print(f"  {sweep_param}={{v}}  train={{tr_auc:.4f}}  val={{val_auc:.4f}}  "
                          f"gap={{tr_auc-val_auc:+.4f}}  ({{fit_sec:.1f}}s)")

        sweep_df = pd.DataFrame(sweep_rows).sort_values("val_auc", ascending=False)
        sweep_df
    """))

    # -------- 6. Final --------
    cells.append(_md(f"""
        # 6. Final Pipeline & Logging

        Refit the chosen configuration on the *full* training set and register
        it as **`{family}_Fraud_Pipeline`** in the MLflow Model Registry.
        Cross-validate with both StratifiedKFold and TimeSeriesSplit to expose
        any temporal-leakage gap.
    """))

    final_kwargs = _kw_repr(cfg["final_kwargs"])
    cells.append(_code(f"""
        FINAL_KW = dict({final_kwargs})
        final_pipe = make_pipeline_with_subset(
            BEST_FS_KEEP,
            cleaner_kwargs=BEST_CLEANING_KW,
            fe_kwargs=BEST_FE_KW,
            model_kwargs=FINAL_KW,
        )
        print("Final config:", FINAL_KW)
    """))

    cells.append(_code(f"""
        with log_run(
            "cv_v1_stratified_5fold",
            params={{"cv": "StratifiedKFold", "n_splits": 5,
                     **{{k: str(v) for k, v in FINAL_KW.items()}}}},
            tags={{"model_family": "{family}", "stage": "cv"}},
        ):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            m = cross_validate_auc(final_pipe, X_train_s, y_train_s, cv=skf)
            log_metrics_dict(m)
            print("Stratified KFold:  val_auc =", m["val_auc"], "+/-", m["val_auc_std"])

        with log_run(
            "cv_v2_timeseries_5fold",
            params={{"cv": "TimeSeriesSplit", "n_splits": 5,
                     **{{k: str(v) for k, v in FINAL_KW.items()}}}},
            tags={{"model_family": "{family}", "stage": "cv"}},
        ):
            tss = TimeSeriesSplit(n_splits=5)
            m = cross_validate_auc(final_pipe, X_train_s, y_train_s, cv=tss)
            log_metrics_dict(m)
            print("TimeSeriesSplit :  val_auc =", m["val_auc"], "+/-", m["val_auc_std"])
    """))

    cells.append(_code(f"""
        with log_run(
            "final_pipeline_full_train",
            params={{
                **{{k: str(v) for k, v in FINAL_KW.items()}},
                "cleaning":       best_cleaning,
                "fe":             best_fe_name,
                "feature_select": best_fs_name,
                "trained_on":     "full_train",
            }},
            tags={{"model_family": "{family}", "stage": "final",
                    "purpose": "register_in_model_registry"}},
        ):
            m = evaluate_holdout(final_pipe, X_train_f, y_train_f, X_val, y_val)
            log_metrics_dict(m)
            proba = final_pipe.predict_proba(X_val)[:, 1]
            mlflow.log_figure(plot_roc(y_val, proba, title="Final {family} - holdout ROC"),
                              "roc_final.png")
            mlflow.log_figure(plot_confusion(y_val, (proba >= 0.5).astype(int),
                                              title="Final {family} - confusion @ 0.5"),
                              "confusion_final.png")
            mlflow.sklearn.log_model(
                sk_model=final_pipe,
                artifact_path="pipeline",
                registered_model_name="{family}_Fraud_Pipeline",
            )
            print(f"Registered model.  holdout val_auc = {{m['val_auc']:.4f}}  gap = {{m['gap']:+.4f}}")
    """))

    cells.append(_md(f"""
        ## What we learned (write this up in the README)

        Paste the actual numbers from MLflow:

        - **Best cleaning** for {family}: `{{best_cleaning}}` -- val_auc on subsample
        - **Best FE**: `{{best_fe_name}}`
        - **Best FS**: `{{best_fs_name}}` -- # features kept
        - **Underfit demo**: which run, train_auc, val_auc
        - **Overfit demo**: which run, train_auc, val_auc, gap
        - **CV gap signal**: TimeSeries vs Stratified val_auc difference
        - **Final holdout AUC**: full-data refit number, used in the cross-model comparison
    """))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _factory_notes(family: str) -> str:
    return {
        "DecisionTree":     "- numeric: constant impute (-999), no scaling\n        - categorical: ordinal-encoded (one-hot would slow trees down)",
        "Bagging":          "- numeric: constant impute (-999)\n        - categorical: ordinal-encoded\n        - base estimator: configurable Decision Tree",
        "RandomForest":     "- numeric: constant impute (-999), no scaling\n        - categorical: ordinal-encoded\n        - bagged trees handle high-dim data well",
        "AdaBoost":         "- numeric: constant impute (-999)\n        - categorical: ordinal-encoded\n        - base estimator: shallow Decision Tree",
        "GradientBoosting": "- numeric: **passthrough** (HistGB handles NaN natively)\n        - categorical: ordinal-encoded",
        "XGBoost":          "- numeric: **passthrough** (XGBoost handles NaN natively via sparsity-aware splits)\n        - categorical: ordinal-encoded",
        "MLP":              "- numeric: median impute + StandardScaler (NN needs zero-mean unit-variance)\n        - categorical: one-hot encoded (capped at top-20 categories per column)",
    }[family]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in MODELS.items():
        out = NB_DIR / f"model_experiment_{name}.ipynb"
        nb = build_notebook(name, cfg)
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"  wrote {out.relative_to(ROOT)}  ({len(nb['cells'])} cells)")
    print("done.")


if __name__ == "__main__":
    main()
