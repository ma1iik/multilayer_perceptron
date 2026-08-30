import json
import os

HISTORY_DIR = "data/history"


def run_name(config):
    """Filename describing the run, so configs do not collide."""
    layers = "-".join(str(size) for size in config["layers"])
    name = (f"{layers}_{config.get('optimizer', 'sgd')}"
            f"_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}")
    if config["early_stop"]:
        name += f"_es{config['patience']}"
    return name


def save_history(hist, config, directory=HISTORY_DIR):
    os.makedirs(directory, exist_ok=True)

    # json cannot serialise numpy floats
    metrics = {key: [float(v) for v in values] for key, values in hist.items()}

    best_epoch = min(range(len(metrics["val_loss"])), key=lambda i: metrics["val_loss"][i])
    payload = {
        "config": config,
        "epochs_run": len(metrics["loss"]),
        "final": {key: values[-1] for key, values in metrics.items()},
        "best": {
            "epoch": best_epoch + 1,
            "val_loss": metrics["val_loss"][best_epoch],
            "val_acc": metrics["val_acc"][best_epoch],
        },
        "metrics": metrics,
    }

    path = os.path.join(directory, run_name(config) + ".json")
    with open(path, "w") as fd:
        json.dump(payload, fd, indent=2)

    print(f"Metrics history saved to {path}")
    return path


def load_history(path):
    with open(path) as fd:
        return json.load(fd)


def list_histories(directory=HISTORY_DIR):
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json"))
