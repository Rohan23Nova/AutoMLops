import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Add the AutoMLOps project directory to Python path
PROJECT_DIR = "/Users/rohankumar/Desktop/AutoMLOps_Project"

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main_pipeline import check_and_retrain


def run_automlops_pipeline():
    os.chdir(PROJECT_DIR)

    print("\n========== AUTOMLOPS PROJECT DIRECTORY ==========")
    print(os.getcwd())

    result = check_and_retrain()

    print("\n========== AUTOMLOPS RESULT ==========")
    print(result)

default_args = {
    "owner": "automlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="automlops_retraining",
    default_args=default_args,
    description="Automated drift detection and ML model retraining",
    start_date=datetime(2026, 1, 1),
    schedule="0 0 * * *",
    catchup=False,
    tags=["automlops", "mlops", "retraining"],
) as dag:

    automated_retraining = PythonOperator(
        task_id="check_drift_and_retrain",
        python_callable=run_automlops_pipeline,
    )