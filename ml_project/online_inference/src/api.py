import os

from fastapi import FastAPI
import pandas as pd

from entities import PredictItemInputRequest, ResponseModel
from constants import PREDICTION_REQUEST_COLUMNS
from utils import get_project_root
from predict_model_wrapper import ModelWrapper

app = FastAPI()

modelWrapper = ModelWrapper(os.path.join(get_project_root(),  "models"))


@app.get("/health")
def health_check():
    return {"status": "Ok"}


@app.post("/predict")
def predict(request: PredictItemInputRequest):
    df = pd.DataFrame([[request.age, request.sex, request.cp, request.trestbps, request.chol,
                        request.fbs, request.restecg, request.thalach, request.exang,
                        request.oldpeak, request.slope, request.ca, request.thal]], columns=PREDICTION_REQUEST_COLUMNS)
    prediction = modelWrapper.predict(df).tolist()[0]
    return ResponseModel(prediction)

