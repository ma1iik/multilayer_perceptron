class EarlyStop:
    def __init__(self, patience=30):
        self.best_loss = float('inf')
        self.max_patience = patience
        self.counter = 0

        self.best_weights = None
        self.best_biases = None

    def check(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0

            self.best_weights = [layer.weights.copy() for layer in model.layers]
            self.best_biases  = [layer.biases.copy() for layer in model.layers]
        else: self.counter += 1

        if self.counter >= self.max_patience:
            return True
        return False


    def restore(self, model):
        if self.best_weights is None:
            return
        
        for layer, w, b in zip(model.layers, self.best_weights, self.best_biases):
            layer.weights = w
            layer.biases = b