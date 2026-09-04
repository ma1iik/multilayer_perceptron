import numpy as np

from core.network import Network
from core.losses import loss_function
from phases.plots import plot_learn_curves
from bonus.early_stop import EarlyStop
from bonus.history import save_history
from bonus.metrics import all_metrics, print_report
from bonus.optimizers import make_optimizer

def load_data(path):
    data = np.genfromtxt(path, dtype = float, delimiter=',')

    labels = data[:, 0]
    features = data[:, 1:]
    one_hot_labels = np.array([[1.00, 0.00] if label == 1 else [0.00, 1.00] for label in labels])
    return features, one_hot_labels

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def train_network(epochs, lr, hlayers, loss_name, batch_s, early_stop=False, patience=30,
                  optimizer_name="sgd", seed=None):
    # random unless asked, so repeated runs converge to different solutions
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    np.random.seed(seed)
    print(f"seed: {seed}")
    model = Network(hlayers)
    optimizer = make_optimizer(optimizer_name, lr)

    train_data, train_labels = load_data("data/train_data.csv")
    val_data, val_labels = load_data("data/val_data.csv")

    print(f"x_train shape : {train_data.shape}")
    print(f"x_valid shape : {val_data.shape}")

    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": [],
            "val_precision": [], "val_recall": [], "val_f1": []}

    # bonus, oof by def
    stopper = EarlyStop(patience) if early_stop else None

    width = len(str(epochs))  # zero-pad the epoch counter

    for epoch in range(epochs):

        order = np.random.permutation(len(train_data))

        for i in range(0, len(train_data), batch_s):
            batch_indx = order[i : i + batch_s]
            x_batch = train_data[batch_indx]
            y_batch = train_labels[batch_indx]

            new_data = model.forward(x_batch)
            model.backward(y_batch, new_data)
            optimizer.step(model.layers)

        val_output = model.forward(val_data)
        train_output = model.forward(train_data)
        val_loss = loss_function(val_labels, val_output, loss_name)
        loss = loss_function(train_labels, train_output, loss_name)

        acc = accuracy(train_labels, train_output)
        val_acc = accuracy(val_labels, val_output)

        hist["loss"].append(loss)
        hist["val_loss"].append(val_loss)
        hist["acc"].append(acc)
        hist["val_acc"].append(val_acc)

        # bonus
        val_metrics = all_metrics(val_labels, val_output)
        hist["val_precision"].append(val_metrics["precision"])
        hist["val_recall"].append(val_metrics["recall"])
        hist["val_f1"].append(val_metrics["f1"])

        print(f"epoch {epoch + 1:0{width}d}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}"
              f" - acc: {acc:.4f} - val_acc: {val_acc:.4f}")
        
        if stopper and stopper.check(val_loss, model):
            print(f"early stopping: no improvement for {patience} epochs")
            break

    if stopper:
        stopper.restore(model)
        print(f"restored best weights (val_loss: {stopper.best_loss:.4f})")

    model.save('data/trained_model.npz')

    print("\nvalidation set:")
    print_report(val_labels, model.forward(val_data))
    print()

    # bonus
    save_history(hist, {
        "layers": list(hlayers),
        "learning_rate": lr,
        "batch_size": batch_s,
        "epochs": epochs,
        "loss": loss_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "early_stop": early_stop,
        "patience": patience,
    })

    plot_learn_curves(hist)
