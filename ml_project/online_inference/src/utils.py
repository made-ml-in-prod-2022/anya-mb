import os
import pickle
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ClassificationModel = Union[LogisticRegression, RandomForestClassifier]


def load_object(filepath: str) -> ClassificationModel:
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_object(obj: ClassificationModel, path: str, filename: str):
    filepath = os.path.join(path, filename)

    with open(filepath, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
