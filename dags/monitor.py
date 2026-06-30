import logging
import os
import re
import json
import html as html_mod
import base64
import struct
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping
from config import (
    DB_URL, TABLE_NAME, MLFLOW_URI, EXPERIMENT, MODEL_NAME,
    MODEL_PATH, PREPROCESSOR_PATH,
    TARGET, CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
)

logger = logging.getLogger(__name__)

PREP_PATH    = PREPROCESSOR_PATH
REPORTS_DIR  = '/opt/airflow/reports'

DRIFT_THRESHOLD  = 3
RMSE_DEGRADE_PCT = 0.20

def _decode_bdata_obj(obj):
    if isinstance(obj, dict):
        if 'bdata' in obj:
            dtype   = obj.get('dtype', 'f8')
            fmt_map = {'f8': 'd', 'f4': 'f', 'i4': 'i', 'i8': 'q',
                       'i2': 'h', 'u2': 'H', 'u1': 'B'}
            fmt       = fmt_map.get(dtype, 'd')
            raw_bytes = base64.b64decode(obj['bdata'])
            n         = len(raw_bytes) // struct.calcsize(fmt)
            vals      = list(struct.unpack_from(f'{n}{fmt}', raw_bytes))
            shape     = obj.get('shape')
            if shape:
                if isinstance(shape, str):
                    dims = [int(x.strip()) for x in shape.split(',')]
                else:
                    dims = list(shape)
                if len(dims) == 2:
                    rows, cols = dims
                    vals = [vals[r * cols:(r + 1) * cols] for r in range(rows)]
            return vals
        return {k: _decode_bdata_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bdata_obj(i) for i in obj]
    return obj


def _fix_evidently_html(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    if 'bdata' not in raw:
        logger.info("_fix_evidently_html: no bdata in %s, skipping", path)
        return

    m_var = re.search(
        r'var (evidently_dashboard_\w+)\s*=\s*(\{.*?\});\s*\n',
        raw, re.DOTALL,
    )

    if m_var:
        dash_var  = m_var.group(1)
        dash_json = m_var.group(2)

        if 'bdata' not in dash_json:
            logger.info("_fix_evidently_html: no bdata in dashboard JSON of %s, skipping", path)
            return

        dash       = json.loads(dash_json)
        dash_fixed = _decode_bdata_obj(dash)
        new_js     = f'var {dash_var} = {json.dumps(dash_fixed)};\n'
        new_raw    = raw[:m_var.start()] + new_js + raw[m_var.end():]

    else:
        m_srcdoc = re.search(r'(srcdoc=")(.*?)(")', raw, re.DOTALL)
        if not m_srcdoc:
            logger.warning("_fix_evidently_html: could not locate dashboard in %s", path)
            return

        srcdoc = html_mod.unescape(m_srcdoc.group(2))
        m_var2 = re.search(
            r'var (evidently_dashboard_\w+)\s*=\s*(\{.*?\});\s*\n',
            srcdoc, re.DOTALL,
        )
        if not m_var2:
            logger.warning("_fix_evidently_html: dashboard variable not found in srcdoc of %s", path)
            return

        dash_var  = m_var2.group(1)
        dash_json = m_var2.group(2)

        if 'bdata' not in dash_json:
            logger.info("_fix_evidently_html: no bdata in %s, skipping", path)
            return

        dash           = json.loads(dash_json)
        dash_fixed     = _decode_bdata_obj(dash)
        new_js         = f'var {dash_var} = {json.dumps(dash_fixed)};\n'
        patched_srcdoc = srcdoc[:m_var2.start()] + new_js + srcdoc[m_var2.end():]

        re_encoded = (
            patched_srcdoc
            .replace('&', '&amp;')
            .replace('"', '&quot;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace("'", '&#x27;')
        )
        new_raw = raw[:m_srcdoc.start(2)] + re_encoded + raw[m_srcdoc.end(2):]

    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_raw)

    logger.info("_fix_evidently_html: decoded bdata blobs in %s", path)


def _save_report(report: Report, path: str) -> None:
    """Save an Evidently report and patch it for correct chart rendering."""
    report.save_html(path)
    _fix_evidently_html(path)
    logger.info("_save_report: wrote %s", path)

def _load_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['functioning_day'] == 'Yes'].copy()
    df['date']        = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df                = df.sort_values(['date', 'hour']).reset_index(drop=True)
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df.drop(columns=['date', 'functioning_day'], inplace=True)
    return df


def _fetch_rmse_baseline() -> float:
    """Read the RMSE of the current Production model from the MLflow Registry."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client   = MlflowClient(tracking_uri=MLFLOW_URI)
        versions = client.get_latest_versions(MODEL_NAME, stages=['Production'])
        if not versions:
            logger.warning("No Production model found — RMSE trigger disabled.")
            return float('inf')
        run_id   = versions[0].run_id
        run      = client.get_run(run_id)
        baseline = run.data.metrics.get('rmse')
        if baseline is None:
            logger.warning("Production run %s has no 'rmse' metric — trigger disabled.", run_id)
            return float('inf')
        logger.info("RMSE baseline from Production model (run %s): %.2f", run_id, baseline)
        return float(baseline)
    except Exception as exc:
        logger.warning("Could not fetch RMSE baseline (%s) — trigger disabled.", exc)
        return float('inf')

def run_monitoring(df_train: pd.DataFrame, df_current: pd.DataFrame, preprocessor) -> dict:
    model = joblib.load(MODEL_PATH)

    X_train   = preprocessor.transform(df_train[ALL_FEATURES])
    X_current = preprocessor.transform(df_current[ALL_FEATURES])

    df_train   = df_train.copy()
    df_current = df_current.copy()

    df_train['prediction']   = np.clip(model.predict(X_train),   0, None).tolist()
    df_current['prediction'] = np.clip(model.predict(X_current), 0, None).tolist()

    for col in NUMERIC_FEATURES + [TARGET]:
        df_train[col]   = df_train[col].astype(float)
        df_current[col] = df_current[col].astype(float)

    col_map = ColumnMapping(
        target               = TARGET,
        prediction           = 'prediction',
        numerical_features   = NUMERIC_FEATURES,
        categorical_features = CATEGORICAL_FEATURES,
    )

    report_cols = ALL_FEATURES + [TARGET, 'prediction']
    os.makedirs(REPORTS_DIR, exist_ok=True)

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data = df_train[report_cols],
        current_data   = df_current[report_cols],
        column_mapping = col_map,
    )
    drift_path = os.path.join(REPORTS_DIR, 'data_drift_report.html')
    _save_report(drift_report, drift_path)

    regression_report = Report(metrics=[RegressionPreset()])
    regression_report.run(
        reference_data = df_train[report_cols],
        current_data   = df_current[report_cols],
        column_mapping = col_map,
    )
    perf_path = os.path.join(REPORTS_DIR, 'regression_performance_report.html')
    _save_report(regression_report, perf_path)

    logger.info("Evidently reports saved to '%s/'", REPORTS_DIR)

    drift_result = drift_report.as_dict()['metrics'][0]['result']
    n_features   = drift_result.get('number_of_columns', 0)
    n_drifted    = drift_result.get('number_of_drifted_columns', 0)

    y_true = df_current[TARGET].values
    y_pred = df_current['prediction'].values
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae    = float(mean_absolute_error(y_true, y_pred))
    r2     = float(r2_score(y_true, y_pred))

    logger.info("Drift  : %d/%d features drifted", n_drifted, n_features)
    logger.info("Metrics: RMSE=%.2f  MAE=%.2f  R2=%.4f", rmse, mae, r2)

    return {
        'n_drifted' : n_drifted,
        'n_features': n_features,
        'rmse'      : rmse,
        'mae'       : mae,
        'r2'        : r2,
        'drift_path': drift_path,
        'perf_path' : perf_path,
    }


def monitor_and_trigger() -> str:
    engine = create_engine(DB_URL)
    try:
        df_all = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", con=engine)
    finally:
        engine.dispose()

    df_all     = _load_and_prepare(df_all)
    split_idx  = int(len(df_all) * 0.80)
    df_train   = df_all.iloc[:split_idx]
    df_current = df_all.iloc[split_idx:]
    logger.info("Reference: %d rows | Current: %d rows", len(df_train), len(df_current))

    preprocessor = joblib.load(PREP_PATH)
    summary      = run_monitoring(df_train, df_current, preprocessor)

    mlflow.set_tracking_uri(MLFLOW_URI)
    rmse_baseline    = _fetch_rmse_baseline()
    rmse_degradation = (
        (summary['rmse'] - rmse_baseline) / rmse_baseline
        if rmse_baseline not in (float('inf'), 0.0)
        else 0.0
    )

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='MONITORING_RUN'):
        mlflow.log_metrics({
            'monitor_rmse'        : round(summary['rmse'], 4),
            'monitor_mae'         : round(summary['mae'],  4),
            'monitor_r2'          : round(summary['r2'],   4),
            'n_drifted_features'  : summary['n_drifted'],
            'rmse_degradation_pct': round(rmse_degradation * 100, 2),
        })
        mlflow.log_artifact(summary['drift_path'])
        mlflow.log_artifact(summary['perf_path'])

    trigger_drift = summary['n_drifted'] > DRIFT_THRESHOLD
    trigger_rmse  = rmse_degradation     > RMSE_DEGRADE_PCT

    if trigger_drift or trigger_rmse:
        reasons = []
        if trigger_drift:
            reasons.append(f"{summary['n_drifted']} features drifted (threshold={DRIFT_THRESHOLD})")
        if trigger_rmse:
            reasons.append(f"RMSE degraded {rmse_degradation*100:.1f}% (threshold={RMSE_DEGRADE_PCT*100:.0f}%)")
        logger.warning("Retrain triggered: %s", ' | '.join(reasons))
        return 'retrain'

    logger.info("Model healthy — no retraining required.")
    return 'ok'


def monitor_task(**context) -> str:
    """Airflow PythonOperator entry point."""
    return monitor_and_trigger()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor_and_trigger()
