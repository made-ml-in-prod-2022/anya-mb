from os.path import join
import pickle

import pandas as pd


def read_data(data_filepath: str) -> pd.DataFrame:

    data = pd.read_csv(data_filepath)
    return data


def save_object(obj: object, path: str, filename: str):
    filepath = join(path, filename)

    with open(filepath, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filepath: str) -> object:
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj
