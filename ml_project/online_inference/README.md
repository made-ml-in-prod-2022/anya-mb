## Installation

`pip install -r requirements.txt`

## Tests
```
export PYTHONPATH=src 
pytest
```

## Web
* run webserver

```
export PYTHONPATH=src 
uvicorn src.api:app --host 0.0.0.0 --port 8088
```

* generated documentation is available by url: 
 
```http://localhost:8088/docs```


## Docker
#### build image
```commandline
# run from the folder with Dockerfile
docker build -t ml_on_prod . 
```

### Run docker container from image

```commandline
export PORT=8088 # free local port
docker run -d -p ${PORT}:80 ml_on_prod
```

### Send health check 
```commandline
curl -v localhost:${PORT}/health
```

### Send prediction request
```commandline
sh ./send_predict_request.sh 
```

### push image to docker hub
```commandline
# go to hub.docker.com and create account first!
docker login
DOCKER_LOGIN="anyamb"
docker tag ml_on_prod ${DOCKER_LOGIN}/ml_on_prod
docker push ${DOCKER_LOGIN}/ml_on_prod 
```

### pull image from docker hub and run
```commandline
docker pull ${DOCKER_LOGIN}/ml_on_prod
docker run -d -p ${PORT}:80 ${DOCKER_LOGIN}/ml_on_prod
```


## Criteria HW2
Основная часть

0. Оберните inference вашей модели в rest сервис на FastAPI, должен быть endpoint /predict (3 балла) +

0. Напишите endpoint /health (1 балл), должен возращать 200, если ваша модель готова к работе (такой чек особенно актуален если делаете доп задание про скачивание из хранилища) +

0. Напишите unit тест для /predict (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/) +

0. Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла +

0. Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл) +

0. опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (+2 балла) +

0. напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл) Убедитесь, что вы можете протыкать его скриптом из пункта 3 +

0. проведите самооценку(распишите в реквесте какие пункты выполнили и на сколько баллов, укажите сумму баллов) -- 1 балл +
