from bonus.history import list_histories, load_history
from phases.plots import plot_comparison


def label_for(config):
    layers = "-".join(str(size) for size in config["layers"])
    label = (f"{layers} {config.get('optimizer', 'sgd')}"
             f" lr{config['learning_rate']} bs{config['batch_size']}")
    if config.get("early_stop"):
        label += " es"
    return label


def print_summary(runs, labels):
    print(f"{'model':34} {'epochs':>6} {'best val_loss':>14} {'final':>8} {'val_f1':>8}")
    print("-" * 74)
    for run, label in zip(runs, labels):
        best = run["best"]
        print(f"{label:34} {run['epochs_run']:6d} "
              f"{best['val_loss']:9.4f} (ep{best['epoch']:>3}) "
              f"{run['final']['val_loss']:8.4f} {run['final'].get('val_f1', 0):8.4f}")


def compare_runs():
    paths = list_histories()
    if not paths:
        raise SystemExit("no saved runs yet, train a few models first")

    runs = [load_history(p) for p in paths]
    # best first
    runs.sort(key=lambda r: r["best"]["val_loss"])
    labels = [label_for(run["config"]) for run in runs]

    print(f"Comparing {len(runs)} runs\n")
    print_summary(runs, labels)
    print()
    plot_comparison(runs, labels)
