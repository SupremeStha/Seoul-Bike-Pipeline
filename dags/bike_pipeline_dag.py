import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# All task files live in the SAME dags/ directory — add it to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasks_ingest     import ingest_task
from tasks_validate   import validate_task
from tasks_load_db    import load_db_task
from tasks_preprocess import preprocess_task
from tasks_train      import train_task
from deploy           import deploy_task

default_args = {
    'owner'           : 'mlops',
    'retries'         : 2,
    'retry_delay'     : timedelta(seconds=30),
    'email_on_failure': False,
}

with DAG(
    dag_id       = 'seoul_bike_pipeline',
    description  = 'End-to-end MLOps pipeline for Seoul Bike Sharing Demand',
    default_args = default_args,
    start_date   = datetime(2025, 1, 1),
    schedule     = '@weekly',
    catchup      = False,
    tags         = ['mlops', 'regression', 'bike-sharing'],
) as dag:

    t1_ingest = PythonOperator(
        task_id         = 'ingest',
        python_callable = ingest_task,
    )

    t2_validate = PythonOperator(
        task_id         = 'validate',
        python_callable = validate_task,
    )

    t3_load_db = PythonOperator(
        task_id         = 'load_db',
        python_callable = load_db_task,
    )

    t4_preprocess = PythonOperator(
        task_id         = 'preprocess',
        python_callable = preprocess_task,
    )

    t5_train = PythonOperator(
        task_id         = 'train',
        python_callable = train_task,
    )

    t6_deploy = PythonOperator(
        task_id         = 'deploy',
        python_callable = deploy_task,
    )

    t1_ingest >> t2_validate >> t3_load_db >> t4_preprocess >> t5_train >> t6_deploy 