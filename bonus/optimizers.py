import numpy as np

# smaller rate for adam
DEFAULT_LR = {"sgd": 0.05, "adam": 0.001}


class SGD:
    """Plain gradient descent, every weight gets the same step size."""

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, layers):
        for layer in layers:
            layer.update_params(self.lr)


class Adam:
    """Scales each weight's step by its own recent gradient history."""

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.state = {}

    def slot(self, layer):
        if id(layer) not in self.state:
            self.state[id(layer)] = {
                "mw": np.zeros_like(layer.weights), "vw": np.zeros_like(layer.weights),
                "mb": np.zeros_like(layer.biases), "vb": np.zeros_like(layer.biases),
            }
        return self.state[id(layer)]

    def step(self, layers):
        # once per batch, not layer
        self.t += 1
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t

        for layer in layers:
            state = self.slot(layer)
            for m, v, param, grad in (("mw", "vw", "weights", "dweights"),
                                      ("mb", "vb", "biases", "dbiases")):
                g = getattr(layer, grad)
                state[m] = self.beta1 * state[m] + (1 - self.beta1) * g
                state[v] = self.beta2 * state[v] + (1 - self.beta2) * g ** 2

                move = self.lr * (state[m] / correction1) / (np.sqrt(state[v] / correction2) + self.eps)
                setattr(layer, param, getattr(layer, param) - move)


def make_optimizer(name, learning_rate):
    return Adam(learning_rate) if name == "adam" else SGD(learning_rate)
