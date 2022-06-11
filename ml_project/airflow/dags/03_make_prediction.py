from datetime import timedelta
import os

from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

from airflow import DAG

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(2),
    "email": ["anyamb@yandex.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}


def wait_for_data(cur_date):
    path = f"data/raw/{cur_date}"
    return os.path.exists(path)


HOST_DATA_DIR = '/Users/annabeketova/Yandex.Disk.localized/made_matirials/ml_in_prod/code/ml_project/airflow'
# HOST_DATA_DIR = '/tmp'
os.makedirs(os.path.join(HOST_DATA_DIR, "data"), exist_ok=True)


prod_model = Variable.get("prod_model", deserialize_json=True)

with DAG(
        dag_id="03_make_prediction",
        default_args=default_args,
        description="make new prediction",
        schedule_interval="@daily"
) as dag:
    wait = PythonSensor(
        task_id="wait_for_data",
        python_callable=wait_for_data,
        op_args=["{{ ds }}"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    make_prediction = DockerOperator(
        task_id="make_prediction",
        image="airflow-make-prediction",
        command="--input-datafile /data/raw/{{ ds }}/heart_cleveland_upload.csv --path-to-model " +
                prod_model["path-to-model"] + " --output-dir /data/predictions/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type='bind')]
    )

    wait >> make_prediction
