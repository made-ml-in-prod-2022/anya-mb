import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from constants import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


def split_train_data(
        data: pd.DataFrame, target_column: str, random_state: int, test_size: float
) -> Tuple:
    X = data.drop(columns=target_column)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=random_state,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split_data(input_dir: str, output_dir: str):
    filepath = os.path.join(input_dir, "train_data.csv")
    data = pd.read_csv(filepath)

    X_train, X_test, y_train, y_test = split_train_data(data, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE)

    print(f"path={os.path.join(output_dir, 'X_train.csv')}")
    print(f"shape is {X_train.shape}")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    split_data()
