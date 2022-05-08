import pandas as pd


def read_data(data_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(data_filepath)
    return data

