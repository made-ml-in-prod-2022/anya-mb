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


logger = logging.getLogger(__name__)
# handler = logging.StreamHandler(sys.stdout)
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)


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
    logger.info(f'Start train pipeline with parameters: {cfg}')
    data = read_data(cfg.data_path)
    logger.info(f'data.shape is {data.shape}')

    X_train, X_test, y_train, y_test = split_data(data, cfg.target_column, cfg.random_state, cfg.test_size)
    logger.info(f'X_train.shape is {X_train.shape}, y_train.shape is {y_train.shape}')

    model = instantiate(cfg.model)
    logger.info(f'Model is {model}')

    model.fit(X_train, y_train)
    logger.info('Model fitted')

    save_object(model, cfg.model_path, 'model.pkl')
    logger.info('Model saved')

    metrics_train = get_metrics(X_train, y_train, model)
    metrics_test = get_metrics(X_test, y_test, model)

    metrics = {'train': metrics_train,
               'test': metrics_test}

    logger.info(f'metrics are {metrics}')

    save_metrics(metrics, cfg.model_path)
    logger.info('Metrics saved')
