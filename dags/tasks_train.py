import io
import logging
import os
import pickle
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import redis
from config import REDIS_HOST, REDIS_PORT, MLFLOW_URI, EXPERIMENT, MODEL_NAME, MODEL_PATH

logger = logging.getLogger(__name__)

MODELS = [
    (
        Ridge(alpha=1.0),
        'Ridge_Regression',
        {'alpha': 1.0}
    ),
    (
        RandomForestRegressor(n_estimators=200, max_depth=15,
                              min_samples_leaf=2, n_jobs=-1, random_state=42),
        'Random_Forest',
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 2}
    ),
    (
        XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     n_jobs=-1, random_state=42, verbosity=0),
        'XGBoost',
        {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05}
    ),
]


def _load_array(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def _pull_arrays(r: redis.Redis) -> tuple:
    arrays = {}
    for key in ['X_train', 'X_test', 'y_train', 'y_test']:
        raw = r.get(key)
        if raw is None:
            raise KeyError(f"Redis key '{key}' not found. "
                           "Ensure preprocess_task ran successfully.")
        arrays[key] = _load_array(raw)

    preprocessor = pickle.loads(r.get('preprocessor'))
    return (arrays['X_train'], arrays['X_test'],
            arrays['y_train'], arrays['y_test'], preprocessor)


def train_and_register() -> str:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    X_train, X_test, y_train, y_test, preprocessor = _pull_arrays(r)
    logger.info("Arrays pulled. X_train: %s  X_test: %s", X_train.shape, X_test.shape)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = []

    for model, name, params in MODELS:
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, None)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)

            mlflow.log_params({'model': name, **params})
            mlflow.log_metrics({'rmse': round(rmse, 4),
                                'mae' : round(mae,  4),
                                'r2'  : round(r2,   4)})

            # Log model artifact INSIDE the open run — never reopen a finished run.
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                artifact_path='model',
                signature=signature,
            )

            logger.info("%-25s  RMSE=%.2f  MAE=%.2f  R2=%.4f", name, rmse, mae, r2)
            results.append({
                'name'  : name,
                'model' : model,
                'run_id': run.info.run_id,
                'rmse'  : rmse,
                'mae'   : mae,
                'r2'    : r2,
            })

    best = min(results, key=lambda x: x['rmse'])
    logger.info("Best model: %s (RMSE=%.2f)", best['name'], best['rmse'])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best['model'], MODEL_PATH)
    logger.info("Saved to %s", MODEL_PATH)

    # Register using the run URI — no need to reopen the run at all
    mlflow.register_model(
        model_uri=f"runs:/{best['run_id']}/model",
        name=MODEL_NAME,
    )

    # Get ALL versions across all stages, sort by version number,
    # take the newest — avoids IndexError when stages=['None'] returns empty
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError(f"No versions found for model '{MODEL_NAME}' after registration.")
    latest_version = sorted(versions, key=lambda v: int(v.version))[-1].version

    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest_version,
        stage='Staging', archive_existing_versions=False)
    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest_version,
        stage='Production', archive_existing_versions=True)

    logger.info("'%s' v%s promoted to Production", MODEL_NAME, latest_version)
    return best['name']


def train_task(**context) -> str:
    """Airflow PythonOperator entry point."""
    return train_and_register()
