"""Metrics calculated only from actual held-out labels and model scores."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.common import TaskType


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    values = []
    for index in range(y_true.shape[1]):
        truth, pred = y_true[:, index], y_pred[:, index]
        tn = int(np.sum((truth == 0) & (pred == 0)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        values.append(tn / (tn + fp) if tn + fp else float("nan"))
    return np.asarray(values, dtype=float)


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[y_true == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float | None, list[float | None]]:
    per_class: list[float | None] = []
    for index in range(y_true.shape[1]):
        per_class.append(_binary_auc(y_true[:, index], probabilities[:, index]))
    valid = [value for value in per_class if value is not None]
    return (float(np.mean(valid)) if valid else None), per_class


def _prf(truth: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, f1 = [], [], []
    for index in range(truth.shape[1]):
        actual, output = truth[:, index], predicted[:, index]
        tp = int(np.sum((actual == 1) & (output == 1)))
        fp = int(np.sum((actual == 0) & (output == 1)))
        fn = int(np.sum((actual == 1) & (output == 0)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.asarray(precision), np.asarray(recall), np.asarray(f1)


def _binary_confusion(truth: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    return [[tn, fp], [fn, tp]]


def evaluate_predictions(
    y_true: np.ndarray | list,
    probabilities: np.ndarray | list,
    classes: list[str],
    task_type: TaskType,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return JSON-serializable class and aggregate metrics.

    ROC-AUC is `None` for labels lacking either positive or negative held-out
    examples. This is a data-coverage limitation, not a score of zero.
    """
    scores = np.asarray(probabilities, dtype=float)
    if task_type == "multilabel":
        truth = np.asarray(y_true, dtype=int)
        if truth.ndim != 2 or scores.shape != truth.shape or scores.shape[1] != len(classes):
            raise ValueError("Multilabel truth/scores must be matching N x number-of-classes arrays")
        predicted = (scores >= threshold).astype(int)
        auc_macro, auc_by_class = _safe_auc(truth, scores)
        precision, recall, f1 = _prf(truth, predicted)
        specificity = _specificity(truth, predicted)
        per_class = {}
        for index, name in enumerate(classes):
            per_class[name] = {
                "precision": float(precision[index]),
                "recall_sensitivity": float(recall[index]),
                "specificity": None if np.isnan(specificity[index]) else float(specificity[index]),
                "f1": float(f1[index]),
                "roc_auc": auc_by_class[index],
                "support_positive": int(truth[:, index].sum()),
            }
        return {
            "task_type": task_type,
            "threshold": threshold,
            "subset_accuracy": float(np.mean(np.all(truth == predicted, axis=1))),
            "precision_macro": float(np.mean(precision)),
            "recall_macro": float(np.mean(recall)),
            "f1_macro": float(np.mean(f1)),
            "roc_auc_macro": auc_macro,
            "per_class": per_class,
            "confusion_matrices": {
                name: _binary_confusion(truth[:, index], predicted[:, index])
                for index, name in enumerate(classes)
            },
        }

    truth = np.asarray(y_true, dtype=int).reshape(-1)
    if scores.ndim != 2 or scores.shape != (len(truth), len(classes)):
        raise ValueError("Multiclass probabilities must be an N x number-of-classes array")
    if np.any((truth < 0) | (truth >= len(classes))):
        raise ValueError("Multiclass target index is outside the supplied class list")
    predicted = np.argmax(scores, axis=1)
    auc_macro, auc_by_class = _safe_auc(
        np.eye(len(classes), dtype=int)[truth], scores
    )
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, output in zip(truth, predicted):
        if actual < 0 or actual >= len(classes):
            raise ValueError(f"Target class index {actual} is outside class list")
        matrix[actual, output] += 1
    per_class = {}
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        per_class[name] = {
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall_sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "roc_auc_ovr": auc_by_class[index],
            "support": int(np.sum(truth == index)),
        }
    return {
        "task_type": task_type,
        "accuracy": float(np.mean(truth == predicted)),
        "precision_macro": float(np.mean([entry["precision"] for entry in per_class.values()])),
        "recall_macro": float(np.mean([entry["recall_sensitivity"] for entry in per_class.values()])),
        "f1_macro": float(np.mean([entry["f1"] for entry in per_class.values()])),
        "roc_auc_macro_ovr": auc_macro,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_order": classes,
    }

