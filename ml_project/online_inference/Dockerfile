# description is here: https://fastapi.tiangolo.com/deployment/docker/#dockerfile
FROM python:3.9

WORKDIR /code/src

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./models /code/models

COPY ./src /code/src

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]