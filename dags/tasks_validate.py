import logging
import pandas as pd
import pyarrow as pa
import redis
import great_expectations as gx
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from config import REDIS_HOST, REDIS_PORT, REDIS_KEY_RAW, REDIS_KEY_VALIDATED

logger = logging.getLogger(__name__)

REDIS_KEY_IN  = REDIS_KEY_RAW
REDIS_KEY_OUT = REDIS_KEY_VALIDATED

EXPECTED_COLUMNS = [
    'date', 'rented_bike_count', 'hour',
    'temperature_c', 'humidity_pct', 'wind_speed_ms',
    'visibility_10m', 'dew_point_temp_c', 'solar_radiation_mj',
    'rainfall_mm', 'snowfall_cm', 'seasons', 'holiday', 'functioning_day',
]


class ValidationError(Exception):
    pass


def run_validation(df: pd.DataFrame) -> None:
    # Use gx.get_context() which works correctly in this container setup
    context = gx.get_context()

    suite = ExpectationSuite(expectation_suite_name='bike_data_suite')

    # Column order
    suite.add_expectation(ExpectationConfiguration(
        expectation_type='expect_table_columns_to_match_ordered_list',
        kwargs={'column_list': EXPECTED_COLUMNS}
    ))

    # Not null checks
    for col in EXPECTED_COLUMNS:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type='expect_column_values_to_not_be_null',
            kwargs={'column': col}
        ))

    # Range checks
    range_checks = [
        ('hour',              0,   23),
        ('rented_bike_count', 0,   None),
        ('humidity_pct',      0,   100),
        ('temperature_c',    -30,  50),
        ('wind_speed_ms',     0,   None),
        ('rainfall_mm',       0,   None),
        ('snowfall_cm',       0,   None),
    ]
    for col, mn, mx in range_checks:
        kwargs = {'column': col}
        if mn is not None:
            kwargs['min_value'] = mn
        if mx is not None:
            kwargs['max_value'] = mx
        suite.add_expectation(ExpectationConfiguration(
            expectation_type='expect_column_values_to_be_between',
            kwargs=kwargs
        ))

    # Set checks
    set_checks = [
        ('seasons',         ['Spring', 'Summer', 'Autumn', 'Winter']),
        ('holiday',         ['Holiday', 'No Holiday']),
        ('functioning_day', ['Yes', 'No']),
    ]
    for col, value_set in set_checks:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type='expect_column_values_to_be_in_set',
            kwargs={'column': col, 'value_set': value_set}
        ))

    validator = context.sources.pandas_default.read_dataframe(df)
    validator.expectation_suite = suite
    result = validator.validate()

    failed = [r for r in result.results if not r.success]
    if failed:
        messages = []
        for r in failed:
            exp_type = r.expectation_config.expectation_type
            col      = r.expectation_config.kwargs.get(
                       'column', r.expectation_config.kwargs.get('column_list', ''))
            messages.append(f"  FAIL — {exp_type} on '{col}'")
            logger.error("FAIL: %s on '%s'", exp_type, col)
        raise ValidationError(
            f"{len(failed)} expectation(s) failed — DAG halted.\n" +
            "\n".join(messages))

    logger.info("All %d expectations passed.", len(result.results))


def validate_and_push() -> str:
    r   = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    raw = r.get(REDIS_KEY_IN)
    if raw is None:
        raise KeyError(f"Redis key '{REDIS_KEY_IN}' not found.")

    df = pa.ipc.open_stream(pa.py_buffer(raw)).read_all().to_pandas()
    logger.info("Pulled from Redis: %d rows", len(df))

    run_validation(df)

    table  = pa.Table.from_pandas(df, preserve_index=False)
    sink   = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    r.set(REDIS_KEY_OUT, sink.getvalue().to_pybytes())
    logger.info("Pushed validated data to Redis key '%s'", REDIS_KEY_OUT)
    return REDIS_KEY_OUT


def validate_task(**context) -> str:
    """Airflow PythonOperator entry point."""
    return validate_and_push()
