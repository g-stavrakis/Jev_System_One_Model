import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
)


def evaluate_model(
    y_true: list,
    y_pred: list,
    latencies: list[float] | None = None,
    costs: list[float] | None = None,
    labels: list | None = None,
) -> dict:
    """
    Evaluate classification performance, latency, and cost.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        latencies: Per-request latency in seconds.
        costs: Per-request cost in USD.
        labels: All possible classification labels, including
                classes potentially absent from y_true.

    Returns:
        Dictionary containing evaluation metrics.
    """
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")

    if latencies is not None and len(latencies) != len(y_true):
        raise ValueError("latencies must match the number of samples.")

    if costs is not None and len(costs) != len(y_true):
        raise ValueError("costs must match the number of samples.")

    if labels is None:
        labels = list(dict.fromkeys([*y_true, *y_pred]))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics = {
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": np.mean(precision),
        "macro_recall": np.mean(recall),
        "macro_f1": np.mean(f1),
        "weighted_f1": np.average(
            f1, weights=support
        ) if sum(support) else 0.0,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }

    if latencies is not None:
        latencies = np.asarray(latencies, dtype=float)

        metrics.update({
            "latency_mean_s": float(np.mean(latencies)),
            "latency_p50_s": float(np.percentile(latencies, 50)),
            "latency_p95_s": float(np.percentile(latencies, 95)),
            "latency_p99_s": float(np.percentile(latencies, 99)),
        })

    if costs is not None:
        costs = np.asarray(costs, dtype=float)

        metrics.update({
            "total_cost_usd": float(np.sum(costs)),
            "cost_per_1000_usd": float(np.mean(costs) * 1000),
        })

    return metrics