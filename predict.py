
import numpy as np
from core.network import Network
from core.losses import loss_function
from train import load_data

def make_pred():
    model = Network.load()

    val_data, val_labels = load_data("data/val_data.csv")

    output = model.forward(val_data)
    predictions = np.argmax(output, axis=1)
    true_indices = np.argmax(val_labels, axis=1)
    print(predictions)
    print(true_indices)

    matches = (predictions == true_indices)
    accur = np.mean(matches) * 100

    loss = loss_function(val_labels, output)
    print(f"The accuracy is {accur}%!")
    print(f"Binary cross-entropy on the validation set: {loss:.4f}")
