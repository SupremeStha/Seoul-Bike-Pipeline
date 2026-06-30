#  Redis
REDIS_HOST = 'myredis'
REDIS_PORT = 6379
REDIS_DB   = 0

# Redis keys passed between pipeline stages
REDIS_KEY_RAW       = 'raw_bike_data'
REDIS_KEY_VALIDATED = 'validated_bike_data'

#  MariaDB 
DB_USER     = 'mariadbuser'
DB_PASSWORD = 'Sunway%40123'   # @ encoded as %40 in connection URL
DB_HOST     = 'mymcs'
DB_PORT     = 3306
DB_NAME     = 'bikedb'
TABLE_NAME  = 'fact_bike_rentals'
DB_URL      = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

#  MLflow 
MLFLOW_URI   = 'file:///opt/airflow/mlruns'
EXPERIMENT   = 'seoul_bike_rental_experiment'
MODEL_NAME   = 'BikeRentalModel'
MODEL_PATH   = '/opt/airflow/models/model.pkl'
PREPROCESSOR_PATH = '/opt/airflow/models/preprocessor.pkl'

#  Features 
TARGET               = 'rented_bike_count'
CATEGORICAL_FEATURES = ['seasons', 'holiday']
NUMERIC_FEATURES     = [
    'hour', 'temperature_c', 'humidity_pct', 'wind_speed_ms',
    'visibility_10m', 'dew_point_temp_c', 'solar_radiation_mj',
    'rainfall_mm', 'snowfall_cm', 'month', 'day_of_week',
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

#  Valid categorical values 
VALID_SEASONS  = ['Spring', 'Summer', 'Autumn', 'Winter']
VALID_HOLIDAYS = ['Holiday', 'No Holiday']
