from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    response = client.post("/predict", json={
        "age": 65,
        "sex": 1,
        "cp": 0,
        "trestbps": 138,
        "chol": 282,
        "fbs": 1,
        "restecg": 2,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 10.4,
        "slope": 1,
        "ca": 1,
        "thal": 0
    })

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_incomplete_request():
    response = client.post("/predict", json={
        "age": 65,
        "sex": 1,
        "cp": 0,
        "trestbps": 138,
        "chol": 282,
        "fbs": 1,
        "restecg": 2,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 10.4,
        "slope": 1,
        "ca": 1,
    })

    assert 400 <= response.status_code < 500


def test_invalid_request():
    response = client.post("/predict", json={
        "random_field": "value",
    })

    assert 400 <= response.status_code < 500
