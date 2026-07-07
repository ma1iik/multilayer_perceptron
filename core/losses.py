import numpy as np

def binary_cross_entropy(y_true, y_pred):
    loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return - np.mean(loss)

def categorical_cross_entropy(y_true, y_pred):
    return - np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def loss_function(y_true, y_pred, name="binaryCrossentropy"):
    # Stop y_pred from ever being exactly 0.0 or 1.0
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    if name == "binaryCrossentropy":
        return binary_cross_entropy(y_true, y_pred)
    if name == "categoricalCrossentropy":
        return categorical_cross_entropy(y_true, y_pred)
    raise ValueError(f"Unknown loss function: {name}")
