import logging
import sys
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

from utils import read_data, save_object


@dataclass
class TrainingParams:
    model: dict
    random_state: int
    model_path: str
    test_size: float
    data_path: str
    target_column: str
    log_path: str


def split_data(
    data, target_column, random_state, test_size
):
    X = data.drop(columns=target_column)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=random_state,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test


def get_metrics(X, y, model) -> Dict[str, float]:
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, predictions)
    return {'accuracy': accuracy,
            'roc_auc': roc_auc}


def save_metrics(metrics: dict, models_path: str):
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

    save_object(model, cfg.model_path, 'model.pkl')

    metrics_train = get_metrics(X_train, y_train, model)
    metrics_test = get_metrics(X_test, y_test, model)

    metrics = {'train': metrics_train,
               'test': metrics_test}

    save_metrics(metrics, cfg.model_path)
    print('Metrics saved')
