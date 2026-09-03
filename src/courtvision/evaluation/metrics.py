"""Metrics for win-probability models.

A win-probability model is judged on two distinct things:
  - discrimination: does it rank games correctly? (accuracy, AUC)
  - calibration: when it says 70%, does the home team win ~70%? (Brier, log loss)
Both are reported against the naive "always pick home" baseline so lift is explicit.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def evaluate(y_true, y_prob, threshold=0.5):
    """Score predicted home-win probabilities against actual outcomes."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
    }


def baseline_scores(y_true, home_win_rate):
    """The "always pick the home team" baseline.

    Accuracy comes from always predicting a home win. For the probabilistic
    metrics we use a constant probability (the training home-win rate) rather
    than 1.0, since predicting 1.0 gives infinite log loss on any home loss.
    """
    y_true = np.asarray(y_true)
    const_prob = np.full(len(y_true), home_win_rate)
    return {
        "accuracy": accuracy_score(y_true, np.ones(len(y_true), dtype=int)),
        "log_loss": log_loss(y_true, const_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, const_prob),
        "auc": 0.5,  # a constant prediction cannot discriminate
    }


def calibration_table(y_true, y_prob, n_bins=10):
    """Bin predictions and compare predicted vs actual win rate per bin.

    A calibrated model has predicted ≈ actual in every bin. Returned as rows of
    (bin_label, n_games, mean_predicted, actual_rate) for printing.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if not mask.any():
            continue
        rows.append((f"{lo:.1f}-{hi:.1f}", int(mask.sum()), y_prob[mask].mean(), y_true[mask].mean()))
    return rows


def format_comparison(model_scores, baseline_scores_dict):
    """Render a model-vs-baseline table with the lift on each metric."""
    lines = [f"{'metric':<10}{'baseline':>10}{'model':>10}{'lift':>10}"]
    for key in ("accuracy", "log_loss", "brier", "auc"):
        base, model = baseline_scores_dict[key], model_scores[key]
        # Lower is better for log_loss and brier, so flip the sign there.
        lift = base - model if key in ("log_loss", "brier") else model - base
        lines.append(f"{key:<10}{base:>10.4f}{model:>10.4f}{lift:>+10.4f}")
    return "\n".join(lines)