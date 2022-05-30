import os

from fastapi import FastAPI
import pandas as pd

from online_inference.entities import PredictItemInputRequest
from utils import get_project_root
from online_inference.predict_model_wrapper import ModelWrapper

app = FastAPI()

modelWrapper = ModelWrapper(os.path.join(get_project_root(),  "models"))


@app.get("/health")
def health_check():
    return {}


@app.post("/predict")
def predict(request: PredictItemInputRequest):
    print(request)
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
               "thal"]
    df = pd.DataFrame([[request.age, request.sex, request.cp, request.trestbps, request.chol,
                        request.fbs, request.restecg, request.thalach, request.exang,
                        request.oldpeak, request.slope, request.ca, request.thal]], columns=columns)
    prediction = modelWrapper.predict(df).tolist()[0]
    return {"prediction": prediction}
