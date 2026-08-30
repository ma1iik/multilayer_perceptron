
import os
import numpy as np
from core.network import Network
from core.losses import loss_function
from bonus.metrics import print_report

TEST_SET = "data/data_test.csv"
VAL_SET = "data/val_data.csv"
NORM_PARAMS = "data/norm_params.npz"


def pick_dataset():
    return TEST_SET if os.path.exists(TEST_SET) else VAL_SET


def scale(features, params=NORM_PARAMS):
    try:
        saved = np.load(params)
    except FileNotFoundError:
        raise SystemExit(f"{params} is missing, run 'python main.py --split' first")
    col_min, col_max = saved["col_min"], saved["col_max"]
    # training min/max, not this file's
    return (features - col_min) / (col_max - col_min + 1e-8)


def load_data(path):
    data = np.genfromtxt(path, delimiter=',', dtype=str)

    if data.shape[1] == 32:            # id, M/B, 30 raw features
        labels = np.where(data[:, 1] == 'M', 1.0, 0.0)
        features = scale(data[:, 2:].astype(float))
    elif data.shape[1] == 31:
        rows = data.astype(float)
        labels = rows[:, 0]
        features = rows[:, 1:]
    else:
        raise SystemExit(f"{path} has {data.shape[1]} columns, expected 32 or 31")

    one_hot_labels = np.array([[1.00, 0.00] if label == 1 else [0.00, 1.00] for label in labels])
    return features, one_hot_labels


def make_pred():
    model = Network.load()

    path = pick_dataset()
    test_data, test_labels = load_data(path)

    output = model.forward(test_data)
    loss = loss_function(test_labels, output)

    print(f"Predicted on {len(test_data)} examples from {path}")
    print(f"Binary cross-entropy: {loss:.4f}")
    print_report(test_labels, output)
