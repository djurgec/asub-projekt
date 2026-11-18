"""
Evaluation utilities for binary classification.

The model should return one logit per image (shape [N] or [N, 1]).
ROC-AUC is computed from probabilities, not thresholded predictions.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate standard binary-classification metrics."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same number of elements.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty set.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values.")

    y_pred = (y_prob >= threshold).astype(int)

    # AUC is undefined when the evaluation set contains only one class.
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class ModelEvaluator:
    """Run a PyTorch model over a DataLoader and calculate metrics."""

    def __init__(self, threshold: float = 0.5, device: Optional[str] = None):
        self.threshold = threshold
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @staticmethod
    def _extract_logits(output):
        # Supports a tensor output or common dict/tuple model outputs.
        if isinstance(output, dict):
            if "logits" in output:
                output = output["logits"]
            else:
                raise ValueError("Model output dict must contain a 'logits' key.")
        elif isinstance(output, (tuple, list)):
            output = output[0]

        if not torch.is_tensor(output):
            raise TypeError("Model must return a torch.Tensor of logits.")
        return output.reshape(-1)

    @torch.no_grad()
    def evaluate(self, model, dataloader) -> Dict[str, float]:
        """
        Evaluate a model on a labelled DataLoader.

        Expected batch format: (images, labels) or (images, labels, image_ids).
        Model output: raw logits, not sigmoid probabilities.
        """
        model = model.to(self.device)
        model.eval()

        all_labels = []
        all_probabilities = []

        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Evaluation DataLoader must return (images, labels).")

            images, labels = batch[0], batch[1]
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32).reshape(-1)

            logits = self._extract_logits(model(images))
            if logits.numel() != labels.numel():
                raise ValueError("Model must return exactly one logit per image.")

            probabilities = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

        return calculate_metrics(
            all_labels,
            all_probabilities,
            threshold=self.threshold,
        )


# Simple functional API for callers who already have predictions.
def evaluate_predictions(y_true, y_prob, threshold: float = 0.5):
    return calculate_metrics(y_true, y_prob, threshold)
