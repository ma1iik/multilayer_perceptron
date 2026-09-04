import numpy as np

# malignant is index 0
POSITIVE = 0


def confusion(y_true, y_pred):
    true = np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)

    tp = int(np.sum((pred == POSITIVE) & (true == POSITIVE)))
    fp = int(np.sum((pred == POSITIVE) & (true != POSITIVE)))
    fn = int(np.sum((pred != POSITIVE) & (true == POSITIVE)))
    tn = int(np.sum((pred != POSITIVE) & (true != POSITIVE)))
    return tp, fp, fn, tn


def all_metrics(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0   # malignant calls that were right
    recall = tp / (tp + fn) if tp + fn else 0.0      # real malignant cases we caught
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def print_report(y_true, y_pred):
    m = all_metrics(y_true, y_pred)

    print(f"accuracy  {m['accuracy'] * 100:6.2f}%")
    print(f"precision {m['precision']:7.4f}   share of the malignant calls that were right")
    print(f"recall    {m['recall']:7.4f}   share of the real malignant cases we caught")
    print(f"f1        {m['f1']:7.4f}   precision and recall combined, low if either is")
    print(f"confusion  malignant: {m['tp']} caught, {m['fn']} missed"
          f" | benign: {m['tn']} correct, {m['fp']} false alarms")
    return m
