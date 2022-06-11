FROM airflow-base:latest

WORKDIR /code

COPY preprocess.py /code/preprocess.py

ENTRYPOINT ["python", "/code/preprocess.py"]