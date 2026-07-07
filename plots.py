import matplotlib.pyplot as plt

def plot_learn_curves(hist):
    epochs = range(1, len(hist["loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 5))

    loss_ax.plot(epochs, hist["loss"], label="training loss")
    loss_ax.plot(epochs, hist["val_loss"], label="validation loss")
    loss_ax.set_xlabel("epochs")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Loss")
    loss_ax.legend()

    acc_ax.plot(epochs, hist["acc"], label="training acc")
    acc_ax.plot(epochs, hist["val_acc"], label="validation acc")
    acc_ax.set_xlabel("epochs")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_title("Accuracy")
    acc_ax.legend()

    plt.show()
