import io
import logging
import os
import pickle
import numpy as np
import pandas as pd
import redis
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine
from config import (
    REDIS_HOST, REDIS_PORT, DB_URL, TABLE_NAME,
    TARGET, CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    VALID_SEASONS, VALID_HOLIDAYS, PREPROCESSOR_PATH,
)

logger = logging.getLogger(__name__)

QUERY_COLUMNS = [
    'date', 'rented_bike_count', 'hour',
    'temperature_c', 'humidity_pct', 'wind_speed_ms',
    'visibility_10m', 'dew_point_temp_c', 'solar_radiation_mj',
    'rainfall_mm', 'snowfall_cm', 'seasons', 'holiday', 'functioning_day',
]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(
            categories=[
                ['Spring', 'Summer', 'Autumn', 'Winter'],
                ['No Holiday', 'Holiday'],
            ],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        ), CATEGORICAL_FEATURES),
        ('num', StandardScaler(), NUMERIC_FEATURES),
    ])


def _save_array(arr: np.ndarray) -> bytes:
    # Serialise a numpy array to bytes using np.save (self-describing format)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def run_preprocessing(df: pd.DataFrame) -> dict:
    df = df[df['functioning_day'] == 'Yes'].copy()
    logger.info("After filtering non-functioning days: %d rows", len(df))

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df.drop(columns=['date', 'functioning_day'], inplace=True)

    split_idx = int(len(df) * 0.80)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET].values
    X_test  = test_df[ALL_FEATURES]
    y_test  = test_df[TARGET].values

    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    logger.info("Preprocessing complete. X_train shape: %s", X_train_proc.shape)

    return {
        'X_train'     : X_train_proc,
        'X_test'      : X_test_proc,
        'y_train'     : y_train,
        'y_test'      : y_test,
        'preprocessor': preprocessor,
    }

def preprocess_and_push() -> str:
    # pandas 2.x + SQLAlchemy 1.4 — pass raw DBAPI connection
    engine   = create_engine(DB_URL)
    cols     = ', '.join(QUERY_COLUMNS)
    raw_conn = engine.raw_connection()
    try:
        df = pd.read_sql(f"SELECT {cols} FROM {TABLE_NAME}", con=raw_conn)
    finally:
        raw_conn.close()
    engine.dispose()    
    logger.info("Queried %d rows from MariaDB", len(df))
    # to_sql sometimes returns integer columns as float64 after a MariaDB round-trip.
    # Cast explicitly to avoid downstream type issues.
    df['rented_bike_count'] = df['rented_bike_count'].astype(int)

    result = run_preprocessing(df)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    # Push arrays using np.save — self-describing, no separate shape/dtype keys needed
    for key in ['X_train', 'X_test', 'y_train', 'y_test']:
        r.set(key, _save_array(result[key]))
        logger.info("Pushed '%s' to Redis: shape %s", key, result[key].shape)

    # Push preprocessor for the train task
    r.set('preprocessor', pickle.dumps(result['preprocessor']))

    # Save preprocessor to disk as well
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(result['preprocessor'], f)
    logger.info("Preprocessor saved to %s", PREPROCESSOR_PATH)

    return 'preprocessing_done'

def preprocess_task(**context) -> str:
    """Airflow PythonOperator entry point."""
    return preprocess_and_push()
