import numpy as np
from core.layers import DenseLayer

class Network:
    def __init__(self, layers=(24, 24)):
        self.hlayers = layers

        self.layers = []
        prev = 30
        for size in layers:
            self.layers.append(DenseLayer(prev, size, activation="sigmoid"))
            prev = size
        self.layers.append(DenseLayer(prev, 2, activation="softmax"))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    
    def backward(self, y_true, y_pred):
        error = (y_pred - y_true) / len(y_true)
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error
    
    # !!!
    def save(self, filename="model.npz"):
        params = {}

        params["topology"] = self.hlayers
        
        for i, layer in enumerate(self.layers):
            params[f"w{i}"] = layer.weights
            params[f"b{i}"] = layer.biases
            
        np.savez(filename, **params)
        print(f"Network safely saved to {filename}!")

    @classmethod
    def load(cls, filename="data/trained_model.npz"):
        saved = np.load(filename)
        model = cls(saved["topology"].tolist())

        for i, layer in enumerate(model.layers):
            layer.weights = saved[f"w{i}"]
            layer.biases = saved[f"b{i}"]
            
        print("Network loaded!")
        return model