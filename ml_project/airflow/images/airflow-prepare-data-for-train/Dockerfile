FROM airflow-base:latest

WORKDIR /code

COPY copy_data.sh /code/copy_data.sh

ENTRYPOINT ["sh", "copy_data.sh"]