"""
Phase 1 connection test — paste this into a Kaggle Notebook cell.

This script is NOT meant to be committed/run as-is in this repo. It is a
reference snippet you copy into Kaggle to verify that:

    1. The three Kaggle Secrets (MLFLOW_TRACKING_URI / USERNAME / PASSWORD)
       are correctly set.
    2. Your Kaggle notebook has internet access enabled.
    3. DagsHub accepts the credentials and a run shows up in the
       MLflow UI (https://dagshub.com/<user>/<repo>.mlflow).

If you see a green run named `connection_test` in the DagsHub MLflow UI
after running this, Phase 1 is complete.

----------------------------------------------------------------------
On Kaggle:
    * Notebook → Settings (right sidebar) → Internet: ON
    * Add-ons → Secrets → enable for this notebook:
        MLFLOW_TRACKING_URI
        MLFLOW_TRACKING_USERNAME
        MLFLOW_TRACKING_PASSWORD
    * Then paste the code below into a cell and Run.
----------------------------------------------------------------------
"""

# Cell 1 -- install MLflow on Kaggle (only mlflow needs installing,
# everything else is preinstalled).
# ! pip install -q mlflow==2.10.2

# Cell 2 -- imports and config
import os
from kaggle_secrets import UserSecretsClient
import mlflow

secrets = UserSecretsClient()
os.environ["MLFLOW_TRACKING_URI"]      = secrets.get_secret("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.get_secret("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.get_secret("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
print("Tracking URI set to:", mlflow.get_tracking_uri())

# Cell 3 -- create a smoke-test experiment + run
mlflow.set_experiment("Connection_Test")

with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("hello",  "world")
    mlflow.log_metric("dummy_metric", 0.42)
    mlflow.set_tag("phase",   "1")
    mlflow.set_tag("purpose", "verify dagshub connection")
    print("Logged a test run. Check the DagsHub MLflow UI now.")
