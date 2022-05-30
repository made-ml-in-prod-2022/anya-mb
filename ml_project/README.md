## Installation

`pip install -r requirements.txt`

## Usage

Random Forest Classifier:

* train: `python3 src/main.py mode=train model=rf`

* predict: `python3 src/main.py mode=predict model=rf`

Logistic Regression:

* train: `python3 src/main.py mode=train model=logreg`

* predict: `python3 src/main.py mode=predict model=logreg`

If it's needed to run on different data, change 'data_path' (in config/mode/predict.yaml). 

Model can be changed with 'model_path' parameter in config/general/general.yaml.

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


## Criteria HW1

Критерии (указаны максимальные баллы, по каждому критерию ревьюер может поставить баллы частично):

0. В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл) +

0. В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл) +

0. Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл) +

0. Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла) +

0. Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла) +

0. Проект имеет модульную структуру (2 балла) +

0. Использованы логгеры (2 балла) +

0. Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла) +/- (частично 1 из 3)

0. Для тестов генерируются синтетические данные, приближенные к реальным (2 балла) +

0. Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла) +

0. Используются датаклассы для сущностей из конфига, а не голые dict (2 балла) +

0. Напишите кастомный трансформер и протестируйте его (3 балла) https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156 -

0. В проекте зафиксированы все зависимости (1 балл) +

0. Настроен CI для прогона тестов, линтера на основе github actions (3 балла). Пример с пары: https://github.com/demo-ml-cicd/ml-python-package +

PS: Можно использовать cookiecutter-data-science https://drivendata.github.io/cookiecutter-data-science/ , но поудаляйте папки, в которые вы не вносили изменения, чтобы не затруднять ревью

Дополнительные баллы=)

* Используйте hydra для конфигурирования (https://hydra.cc/docs/intro/) - 3 балла +

Самооценка: 24-28 баллов. Тесты есть на отдельные функции и модули, но нет на train_pipeline и predict_pipeline. Нет кастомного трансформера. Остальное все присутствует. Это для меня первый такой проект, поэтому может быть что-то не так сделано, но я старалась разобраться.