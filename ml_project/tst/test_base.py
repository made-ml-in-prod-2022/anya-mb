from os.path import join, isfile
import py

import pandas as pd
import numpy as np

from utils import save_object, load_object, read_data
from train import split_data, save_metrics
from predict import save_predictions

DATA_PATH = 'data/heart_cleveland_upload.csv'
TMP_DIR = py.path.local('./src/test_data')
TMP_FILE = join(TMP_DIR, 'tmp_file')
FAKE_DATA_FILEPATH = join(TMP_DIR, 'fake_data.csv')
FAKE_DATA_LEN = 100


def make_fake_data(length=FAKE_DATA_LEN):
    data = pd.read_csv(DATA_PATH)
    sample = {}

    # sample fake data by columns
    for col in data.columns:
        sample[col] = np.random.choice(data[col].values, size=length, replace=True)

    pd.DataFrame(data=sample).to_csv(FAKE_DATA_FILEPATH, index=False)


def read_fake_data() -> pd.DataFrame:
    fake_data = pd.read_csv(FAKE_DATA_FILEPATH)
    return fake_data


def test_read_data_read_correct_file():
    read = read_data(DATA_PATH)
    assert read.shape == (297, 14), (
        'Datapath is not correct'
    )


def test_save_object_and_load_object_are_the_same():
    expected_str = 'cerwvtrwva'
    save_object(expected_str, TMP_DIR, 'tmp_file')

    loaded = load_object(join(TMP_DIR, 'tmp_file'))

    assert expected_str == loaded, (
        'Saved and loaded object is not as initial'
    )


def test_split_data():
    make_fake_data(10)
    data = read_fake_data()
    n_feature_columns = data.shape[1] - 1
    target_column = 'condition'
    random_state = 5
    test_size = 0.3
    X_train, X_test, y_train, y_test = split_data(data, target_column, random_state, test_size)
    assert X_train.shape == (7, n_feature_columns), (
        f'X_train.shape is {X_train.shape}, expected {(7, n_feature_columns)}'
    )
    assert X_test.shape == (3, n_feature_columns), (
        f'X_test.shape is {X_test.shape}, expected {(3, n_feature_columns)}'
    )
    assert y_train.shape == (7, ), (
        f'y_train.shape is {y_train.shape}, expected (7, )'
    )
    assert y_test.shape == (3, ), (
        f'y_test.shape is {y_test.shape}, expected (3, )'
    )


def test_save_metrics():
    metrics = {'accuracy': 0.9,
               'roc_auc': 0.85}
    file_path = join(TMP_DIR, 'metrics.json')
    save_metrics(metrics, TMP_DIR)
    assert isfile(file_path), (
        'save_metrics does not save to the file'
    )


def test_save_predictions():
    predictions = np.array([1, 0, 1])
    file_path = join(TMP_DIR, 'predictions.csv')
    save_predictions(predictions, TMP_DIR)
    assert isfile(file_path), (
        'save_predictions does not save to the file'
    )
