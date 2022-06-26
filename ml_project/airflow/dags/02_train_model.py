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
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

HOST_DATA_DIR = '/Users/annabeketova/Yandex.Disk.localized/made_matirials/ml_in_prod/code/ml_project/airflow'
# HOST_DATA_DIR = '/tmp'
os.makedirs(os.path.join(HOST_DATA_DIR, "data"), exist_ok=True)


def _wait_for_data(cur_date):
    path = f"data/raw/{cur_date}"
    return os.path.exists(path)


with DAG(
        dag_id="02_train_model",
        default_args=default_args,
        description="train_new_model",
        schedule_interval="@weekly"
) as dag:
    wait = PythonSensor(
        task_id="wait_for_data",
        python_callable=_wait_for_data,
        op_args=["{{ ds }}"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    download_data = DockerOperator(
        task_id="copy_data",
        image="airflow-prepare-data-for-train",
        command="/data/raw/{{ ds }} /data/processed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
    )

    split_data = DockerOperator(
        task_id="split_data",
        image="airflow-split-train-data",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
    )

    preprocess_data = DockerOperator(
        task_id="preprocess_data",
        image="airflow-preprocess-data",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }} --output-model-dir "
                "/data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
    )


    def generate_tasks_for_model(model_name: str):
        train_task = DockerOperator(
            task_id=f"train_{model_name}",
            image="airflow-train-model",
            command="--input-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }} --model-name " + model_name,
            network_mode="bridge",
            do_xcom_push=False,
            mount_tmp_dir=False,
            mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
        )

        evaluate_task = DockerOperator(
            task_id=f"evaluate_{model_name}",
            image="airflow-evaluate-model",
            command="--input-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }} --model-name " + model_name,
            network_mode="bridge",
            do_xcom_push=False,
            mount_tmp_dir=False,
            mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
        )

        train_task >> evaluate_task

        return [train_task, evaluate_task]


    compare_models = DockerOperator(
        task_id="compare_models",
        image="airflow-compare-models",
        command="--model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f"{HOST_DATA_DIR}/data", target="/data", type="bind")]
    )

    models_tasks = [
        generate_tasks_for_model("rf"),
        generate_tasks_for_model("logreg")
    ]

    wait >> download_data >> split_data >> preprocess_data

    for tasks_about_model in models_tasks:
        preprocess_data >> tasks_about_model[0]
        tasks_about_model[-1] >> compare_models

