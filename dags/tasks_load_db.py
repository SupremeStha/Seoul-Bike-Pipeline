import logging
import pandas as pd
import pyarrow as pa
import redis
from sqlalchemy import create_engine, text
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_KEY_VALIDATED,
    DB_URL, DB_NAME, TABLE_NAME,
)

logger = logging.getLogger(__name__)

REDIS_KEY_IN = REDIS_KEY_VALIDATED


def get_engine():
    """Create and return a SQLAlchemy engine for MariaDB."""
    return create_engine(DB_URL)


def _sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise DataFrame column names once — used by both DDL builder and insert."""
    df.columns = [
        c.replace(' ', '_').replace('(', '').replace(')', '')
        for c in df.columns
    ]
    return df


def _build_create_table_sql(df: pd.DataFrame, table_name: str) -> str:
    type_map = {
        'int64':   'BIGINT',
        'int32':   'INT',
        'float64': 'DOUBLE',
        'float32': 'FLOAT',
        'bool':    'TINYINT(1)',
        'object':  'VARCHAR(255)',
    }

    col_defs = []
    for col, dtype in df.dtypes.items():
        # Detect date columns explicitly for a cleaner schema
        if 'date' in col.lower():
            sql_type = 'DATE'
        else:
            sql_type = type_map.get(str(dtype), 'VARCHAR(255)')
        col_defs.append(f"  `{col}` {sql_type}")

    cols_sql = ",\n".join(col_defs)
    return (
        f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n"
        f"{cols_sql}\n"
        f") ENGINE=Columnstore DEFAULT CHARSET=utf8mb4;"
    )


def load_to_mariadb() -> str:
    # Pull from Redis
    r   = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    raw = r.get(REDIS_KEY_IN)
    if raw is None:
        raise KeyError(
            f"Redis key '{REDIS_KEY_IN}' not found. "
            "Ensure validate_task ran successfully."
        )

    df = pa.ipc.open_stream(pa.py_buffer(raw)).read_all().to_pandas()
    logger.info("Pulled from Redis: %d rows x %d columns", *df.shape)

    # Sanitise column names once — used by both DDL builder and insert
    df = _sanitise_columns(df)

    # Convert date columns from DD/MM/YYYY to YYYY-MM-DD 
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            logger.info("Converted date column '%s' to YYYY-MM-DD format.", col)

    # Write to MariaDB ColumnStore 
    engine = get_engine()
    try:
        # Check if table already exists
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl"
            ), {"db": DB_NAME, "tbl": TABLE_NAME})
            table_exists = result.scalar() > 0

        if not table_exists:
            # First run only — CREATE TABLE with ColumnStore engine
            create_sql = _build_create_table_sql(df, TABLE_NAME)
            logger.info("Creating ColumnStore table...\nDDL:\n%s", create_sql)
            with engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.execute(text("COMMIT"))
            logger.info("Table created successfully.")
        else:
            # Table exists — TRUNCATE (does NOT need DDLProc, much safer)
            logger.info("Table exists, truncating for fresh load...")
            with engine.connect() as conn:
                conn.execute(text(f"TRUNCATE TABLE `{TABLE_NAME}`"))
                conn.execute(text("COMMIT"))
            logger.info("Table truncated.")

        df.to_sql(
            name      = TABLE_NAME,
            con       = engine,
            if_exists = 'append',
            index     = False,
            chunksize = 1000,
        )
    finally:
        engine.dispose()

    logger.info("Written %d rows to MariaDB ColumnStore table '%s'", len(df), TABLE_NAME)
    return TABLE_NAME


def load_db_task(**context) -> str:
    """Airflow PythonOperator entry point."""
    return load_to_mariadb()
