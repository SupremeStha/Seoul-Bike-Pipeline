import logging
import os
import pandas as pd
import pyarrow as pa
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_KEY_RAW

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    'Date'                         : 'date',
    'Rented Bike Count'            : 'rented_bike_count',
    'Hour'                         : 'hour',
    'Temperature(\xb0C)'           : 'temperature_c',
    'Humidity(%)'                  : 'humidity_pct',
    'Wind speed (m/s)'             : 'wind_speed_ms',
    'Visibility (10m)'             : 'visibility_10m',
    'Dew point temperature(\xb0C)' : 'dew_point_temp_c',
    'Solar Radiation (MJ/m2)'      : 'solar_radiation_mj',
    'Rainfall(mm)'                 : 'rainfall_mm',
    'Snowfall (cm)'                : 'snowfall_cm',
    'Seasons'                      : 'seasons',
    'Holiday'                      : 'holiday',
    'Functioning Day'              : 'functioning_day',
}

REDIS_KEY  = REDIS_KEY_RAW


def read_and_push(csv_path: str) -> str:
    # Load the CSV, rename columns, parse the date, serialise to Arrow,
    # and push to Redis. Returns the Redis key.
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df.rename(columns=COLUMN_MAP, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    logger.info("Loaded %d rows. Date range: %s to %s",
                len(df), df['date'].min().date(), df['date'].max().date())

    table  = pa.Table.from_pandas(df, preserve_index=False)
    sink   = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.set(REDIS_KEY, sink.getvalue().to_pybytes())
    logger.info("Pushed to Redis key '%s'", REDIS_KEY)

    return REDIS_KEY


def ingest_task(**context) -> str:
    from airflow.models import Variable
    # default_var=None gives a clear error message instead of a generic KeyError
    csv_path = Variable.get("bike_csv_path", default_var=None)
    if not csv_path:
        raise ValueError(
            "Airflow Variable 'bike_csv_path' is not set. "
            "Go to Admin > Variables and add it pointing to SeoulBikeData.csv"
        )
    return read_and_push(csv_path)
