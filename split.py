import csv
import numpy as np


def split_dataset():
    print("Shuffling and splitting data...")

    data = np.genfromtxt(
        'data/data.csv',
        delimiter=',',
        dtype=str,
    )

    # M/B to 1/0
    data[:, 1] = np.where(data[:, 1] == 'M', '1', '0')

    #  data type to float
    data = data.astype(float)

    # random shuffling
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data[shuffled_indices]

    # split into training and validation data
    split_point = int(len(shuffled_data) * 0.8)
    train_data = shuffled_data[:split_point, 1:]
    val_data = shuffled_data[split_point:, 1:]

    # data normalisation (col 0 is the label, cols 1: are the 30 features)
    features = train_data[:, 1:]
    col_min = np.min(features, axis=0) #axis 0 to look accros a col, axis 1 to look across the row
    col_max = np.max(features, axis=0)

    train_data[:, 1:] = (features - col_min) / (col_max - col_min + 1e-8)

    features = val_data[:, 1:]
    val_data[:, 1:] = (features - col_min) / (col_max - col_min + 1e-8)

    np.savetxt('data/train_data.csv', train_data, delimiter=',')
    np.savetxt('data/val_data.csv', val_data, delimiter=',')
