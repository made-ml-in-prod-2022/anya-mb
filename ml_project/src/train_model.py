from dataclasses import dataclass
from typing import Dict
from os.path import join
import json

from hydra.utils import instantiate

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from utils import read_data

MODELS_PATH = '../models'
DATA_PATH = '../data/heart_cleveland_upload.csv'
TARGET_COLUMN = 'condition'

RANDOM_STATE = 91
TEST_SIZE = 0.2
MODEL_NAME = 'RF'


@dataclass
class TrainingParams:
    model: dict
    random_state: int
    model_path: str
    test_size: float
    data_path: str
    target_column: str


def split_data(
    data, target_column, random_state, test_size
):
    X = data.drop(columns=target_column)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=random_state,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test


# def fit_model(
#     model_name, X, y
# ):
#     if model_name == 'LR':
#         model = LogisticRegression(random_state=RANDOM_STATE)
#     elif model_name == 'RF':
#         model = RandomForestClassifier(random_state=RANDOM_STATE)
#     else:
#         raise NotImplementedError()
#
#     model.fit(X, y)
#     return model


def get_metrics(
    X, y, model
) -> Dict[str, float]:
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    return {'accuracy': accuracy,
            'roc_auc': roc_auc}


def save_metrics(
    metrics: dict, models_path: str
):
    metric_path = join(models_path, 'metrics.json')

    with open(metric_path, 'w') as outfile:
        json.dump(metrics, outfile)


def train_pipeline(cfg: TrainingParams):
    data = read_data(cfg.data_path)
    print('Reading data')

    X_train, X_test, y_train, y_test = split_data(data, cfg.target_column, cfg.random_state, cfg.test_size)

    model = instantiate(cfg.model)
    model.fit(X_train, y_train)
    print('Fitting model')
    metrics_train = get_metrics(X_train, y_train, model)
    metrics_test = get_metrics(X_test, y_test, model)

    metrics = {'train': metrics_train,
               'test': metrics_test}

    save_metrics(metrics, cfg.model_path)
    print('Metrics saved')
