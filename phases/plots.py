import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES_PATH = "data/learning_curves.png"
COMPARISON_PATH = "data/model_comparison.png"


def plot_learn_curves(hist, path=CURVES_PATH):
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

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved to {path}")


def plot_comparison(runs, labels, path=COMPARISON_PATH):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(14, 5))

    for run, label in zip(runs, labels):
        metrics = run["metrics"]
        epochs = range(1, len(metrics["val_loss"]) + 1)

        line, = loss_ax.plot(epochs, metrics["val_loss"], label=label)
        # faded: each model's own training loss
        loss_ax.plot(epochs, metrics["loss"], color=line.get_color(), alpha=0.25, linestyle="--")
        acc_ax.plot(epochs, metrics["val_acc"], color=line.get_color(), label=label)

    loss_ax.set_xlabel("epochs")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Validation loss (dashed = training loss)")
    loss_ax.legend(fontsize=8)

    acc_ax.set_xlabel("epochs")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_title("Validation accuracy")
    acc_ax.legend(fontsize=8)

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Model comparison saved to {path}")
