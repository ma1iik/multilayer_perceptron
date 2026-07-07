import numpy as np
from core.activations import sigmoid, softmax

class DenseLayer:
    def __init__(self, input_size, num_neurons, activation="sigmoid"):
        #scale down so  weighted sums don't saturate sigmoid at start (xavier init)
        self.weights = np.random.randn(input_size, num_neurons) * np.sqrt(1 / input_size)
        self.biases = np.zeros((1, num_neurons))
        self.activation = activation;

    def forward(self, inputs):
        self.inputs = inputs

        self.raw_output = np.dot(inputs, self.weights) + self.biases
        if self.activation == "softmax":
            self.output = softmax(self.raw_output)
        else:
            self.output = sigmoid(self.raw_output)

        return self.output
    
    def backward(self, errors):
        if self.activation == "sigmoid":
            derivative = self.output * (1 - self.output)
            errors = errors * derivative
        self.dweights = np.dot(self.inputs.T, errors)
        self.dbiases = np.sum(errors, axis=0, keepdims=True)
        dinputs = np.dot(errors, self.weights.T)
        return dinputs
    
    def update_params(self, learn_rate):
        self.weights -= (learn_rate * self.dweights)
        self.biases -= (learn_rate * self.dbiases)
