import numpy as np

from core.network import Network
from core.losses import loss_function

def load_data(path):
    data = np.genfromtxt(path, dtype = float, delimiter=',')

    labels = data[:, 0]
    features = data[:, 1:]
    one_hot_labels = np.array([[1.00, 0.00] if label == 1 else [0.00, 1.00] for label in labels])
    return features, one_hot_labels

def train_network(epochs, lr, hlayers, loss_name):
    np.random.seed(42)
    model = Network(hlayers)

    train_data, train_labels = load_data("data/train_data.csv")
    val_data, val_labels = load_data("data/val_data.csv")

    for epoch in range(epochs):
        new_data = model.forward(train_data)
        loss = loss_function(train_labels, new_data, loss_name)
        model.backward(train_labels, new_data)
        for layer in model.layers: layer.update_params(lr)

        val_output = model.forward(val_data)
        val_loss = loss_function(val_labels, val_output, loss_name)
        print(f"epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

    model.save('data/trained_model.npz')
