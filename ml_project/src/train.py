import logging
from dataclasses import dataclass
from typing import Dict, Union, Tuple
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
ClassificationModel = Union[LogisticRegression, RandomForestClassifier]


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
    data: pd.DataFrame, target_column: str, random_state: int, test_size: float
) -> Tuple:
    X = data.drop(columns=target_column)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=random_state,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test


def get_metrics(y: pd.DataFrame, predictions, pred_probas) -> Dict[str, float]:
    accuracy = accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, pred_probas)
    return {'accuracy': accuracy,
            'roc_auc': roc_auc}


def save_metrics(metrics: dict, models_path: str):
    metric_path = join(models_path, 'metrics.json')

    with open(metric_path, 'w') as outfile:
        json.dump(metrics, outfile)


def predict(x: pd.DataFrame, model: ClassificationModel) -> Tuple:
    predictions = model.predict(x)
    pred_probas = model.predict_proba(x)
    return predictions, pred_probas


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

    predictions_train, pred_probas_train = predict(X_train, model)
    predictions_test, pred_probas_test = predict(X_test, model)

    logger.info('Predictions generated')

    metrics_train = get_metrics(y_train, predictions_train, pred_probas_train)
    metrics_test = get_metrics(y_test, predictions_test, pred_probas_test)

    metrics = {'train': metrics_train,
               'test': metrics_test}

    logger.info(f'metrics are {metrics}')

    save_metrics(metrics, cfg.model_path)
    logger.info('Metrics saved')
