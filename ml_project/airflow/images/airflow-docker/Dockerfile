FROM apache/airflow:2.3.0-python3.8

RUN pip install --user --no-cache-dir apache-airflow-providers-docker==2.6.0

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages