import json
import logging
from dataclasses import dataclass
from os.path import join
from typing import Dict, Tuple

import pandas as pd
from hydra.utils import instantiate
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from utils import read_data, save_object, ClassificationModel
from feature_transform import ExperimentalTransformer


logger = logging.getLogger(__name__)

MODEL_NAME = 'model.pkl'
METRICS_NAME = 'metrics.json'


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


def get_metrics(target: pd.DataFrame, predictions, pred_probas) -> Dict[str, float]:
    accuracy = accuracy_score(target, predictions)
    roc_auc = roc_auc_score(target, pred_probas[:, 1])
    return {'accuracy': accuracy,
            'roc_auc': roc_auc}


def save_metrics(metrics: dict, models_path: str):
    metric_path = join(models_path, METRICS_NAME)

    with open(metric_path, 'w') as outfile:
        json.dump(metrics, outfile)


def predict(features: pd.DataFrame, model: ClassificationModel) -> Tuple:
    predictions = model.predict(features)
    pred_probas = model.predict_proba(features)
    return predictions, pred_probas


def train_pipeline(cfg: TrainingParams):
    logger.info(f'Start train pipeline with parameters: {cfg}')
    data = read_data(cfg.data_path)
    logger.info(f'data.shape is {data.shape}')

    X_train, X_test, y_train, y_test = split_data(data, cfg.target_column, cfg.random_state, cfg.test_size)
    logger.info(f'X_train.shape is {X_train.shape}, y_train.shape is {y_train.shape}')

    model = instantiate(cfg.model)
    logger.info(f'Model is {model}')

    transformer = ExperimentalTransformer(cfg.model_path)

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    transformer.save_to_file()

    model.fit(X_train_transformed, y_train)
    logger.info('Model fitted')

    save_object(model, cfg.model_path, MODEL_NAME)
    logger.info('Model saved')

    predictions_train, pred_probas_train = predict(X_train_transformed, model)
    predictions_test, pred_probas_test = predict(X_test_transformed, model)

    logger.info('Predictions generated')

    metrics_train = get_metrics(y_train, predictions_train, pred_probas_train)
    metrics_test = get_metrics(y_test, predictions_test, pred_probas_test)

    metrics = {'train': metrics_train,
               'test': metrics_test}

    logger.info(f'metrics are {metrics}')

    save_metrics(metrics, cfg.model_path)
    logger.info('Metrics saved')
