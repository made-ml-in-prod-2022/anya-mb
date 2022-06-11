FROM airflow-base:latest

WORKDIR /code

COPY validate_raw_data.py /code/validate_raw_data.py

ENTRYPOINT ["python", "/code/validate_raw_data.py"]