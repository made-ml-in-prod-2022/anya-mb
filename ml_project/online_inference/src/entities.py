from dataclasses import dataclass

from pydantic import BaseModel, validator


@dataclass
class ResponseModel:
    prediction: float


class PredictItemInputRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: int
    slope: int
    ca: int
    thal: int

    @validator('sex')
    def sex_has_only_2_values(cls, sex):
        if sex not in [0, 1]:
            raise ValueError("Invalid sex value")
        return sex

    @validator("age")
    def age_smoke_validation(cls, age):
        if age < 0 or age > 150:
            raise ValueError("Invalid age value")
        return age
