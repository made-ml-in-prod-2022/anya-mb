import os
from datetime import timedelta

from airflow import DAG

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(2),
    "email": ["anyamb@yandex.ru"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(seconds=10)
}

HOST_DATA_DIR = '/Users/annabeketova/Yandex.Disk.localized/made_matirials/ml_in_prod/code/ml_project/airflow'
# HOST_DATA_DIR = '/tmp'
os.makedirs(os.path.join(HOST_DATA_DIR, "data"), exist_ok=True)


def wait_for_data(cur_date):
    path = f"data/raw/{cur_date}"
    return os.path.exists(path)


with DAG(
        dag_id="01_get_data",
        default_args=default_args,
        description="get new batch of data",
        schedule_interval="@daily"
) as dag:
    download_data = DockerOperator(
        task_id="download_data",
        image="airflow-download-data",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
    )

    wait = PythonSensor(
        task_id="wait_for_data",
        python_callable=wait_for_data,
        op_args=["{{ ds }}"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    validate_data = DockerOperator(
        task_id="validate_downloaded_data",
        image="airflow-validate-downloaded-data",
        command="--path /data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
    )

    download_data >> wait >> validate_data
