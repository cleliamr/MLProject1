import numpy as np


#with open("dataset/y_train.csv") as f:
    #for _ in range(3):
        #print(f.readline())

import numpy as np

import numpy as np
import csv


def _read_csv(path):
    """Reads a CSV file into a NumPy float array, treating empty cells as NaN."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        data = [
            [float(val) if val != "" else np.nan for val in row]
            for row in reader
        ]
    return np.array(data)


def load_csv_data(data_path="dataset", sub_sample=False):
    """
    Loads x_train, y_train, and x_test from CSV files using only NumPy.

    Returns:
        y_train: np.array of shape (N,), labels (-1 or 1)
        x_train: np.array of shape (N, D), features (Id column dropped)
        x_test: np.array of shape (N_test, D), features (Id column dropped)
        train_ids: np.array of shape (N,), the Id column from train
        test_ids: np.array of shape (N_test,), the Id column from test
    """
    x_train_full = _read_csv(f"{data_path}/x_train.csv")
    x_test_full = _read_csv(f"{data_path}/x_test.csv")
    y_train_full = _read_csv(f"{data_path}/y_train.csv")

    train_ids = x_train_full[:, 0].astype(int)
    test_ids = x_test_full[:, 0].astype(int)
    y_ids = y_train_full[:, 0].astype(int)

    # Safety check: make sure x_train and y_train rows are aligned
    assert np.array_equal(train_ids, y_ids), (
        "x_train and y_train Ids do not match in order! "
        "You need to realign them before proceeding."
    )

    x_train = x_train_full[:, 1:]
    x_test = x_test_full[:, 1:]
    y_train = y_train_full[:, 1]

    if sub_sample:
        x_train = x_train[::50]
        y_train = y_train[::50]
        train_ids = train_ids[::50]

    return y_train, x_train, x_test, train_ids, test_ids