import logging
import os
import joblib
import mlflow.sklearn

logger = logging.getLogger(__name__)

MODEL_NAME  = 'BikeRentalModel'
MODEL_STAGE = 'Production'

MODELS_DIR  = '/opt/airflow/models'
MODEL_PATH  = os.path.join(MODELS_DIR, 'model.pkl')
MLFLOW_URI  = 'file:///opt/airflow/mlruns'


def save_model_from_registry() -> str:
    # Pull the Production model from the MLflow Registry and save it to disk.
    os.makedirs(MODELS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from registry: %s", model_uri)

    model = mlflow.sklearn.load_model(model_uri)

    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    return MODEL_PATH


def verify_model(path: str) -> None:
    # Reload the saved model from disk and confirm it is readable
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"model.pkl not found at '{path}' after deploy step.")

    model   = joblib.load(path)
    size_kb = os.path.getsize(path) / 1024
    logger.info("Verified: %s (%.1f KB) at %s",
                type(model).__name__, size_kb, path)


def deploy_task(**context) -> str:
    # Airflow PythonOperator entry point
    path = save_model_from_registry()
    verify_model(path)
    logger.info("Deploy complete — FastAPI container ready to serve.")
    return path
