from os.path import join
from typing import Union
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

ClassificationModel = Union[LogisticRegression, RandomForestClassifier]


def read_data(data_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(data_filepath)
    return data


def save_object(obj: ClassificationModel, path: str, filename: str):
    filepath = join(path, filename)

    with open(filepath, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filepath: str) -> ClassificationModel:
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))