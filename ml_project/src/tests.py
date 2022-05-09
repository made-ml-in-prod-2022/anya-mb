import pytest
from train import read_data
from utils import save_object, load_object
import py
from os.path import join

DATA_PATH = '../data/heart_cleveland_upload.csv'
TMP_DIR = py.path.local('.')


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
