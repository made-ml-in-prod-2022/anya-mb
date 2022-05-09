from os.path import join
import pickle

import pandas as pd


def read_data(data_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(data_filepath)
    return data


def save_object(obj, path, filename):
    filepath = join(path, filename)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_object(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj