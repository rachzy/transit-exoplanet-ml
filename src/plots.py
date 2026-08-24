"""Report figures. All plotting goes through a non-interactive backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .stacking import STACK, reported_models

FIGSIZE = (7.5, 5.0)
DPI = 140


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def calibration_plot(calibration: dict[str, list[dict[str, float]]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name, points in calibration.items():
        if not points:
            continue
        x = [p["mean_predicted"] for p in points]
        y = [p["observed_rate"] for p in points]
        ax.plot(x, y, marker="o", lw=2 if name == STACK else 1,
                alpha=1.0 if name == STACK else 0.6, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed CONFIRMED rate")
    ax.set_title("Calibration on held-out stars (accepted candidates)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def pr_curve_plot(oof: pd.DataFrame, path: Path) -> Path:
    y = oof["y_true"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in reported_models():
        column = f"prob_{name}"
        if column not in oof:
            continue
        precision, recall, _ = precision_recall_curve(y, oof[column].to_numpy(dtype=float))
        ax.plot(recall, precision, lw=2 if name == STACK else 1,
                alpha=1.0 if name == STACK else 0.6, label=name)
    ax.axhline(y.mean(), color="k", ls=":", lw=1, label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall on held-out stars (accepted candidates)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def roc_curve_plot(oof: pd.DataFrame, path: Path) -> Path:
    y = oof["y_true"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in reported_models():
        column = f"prob_{name}"
        if column not in oof:
            continue
        fpr, tpr, _ = roc_curve(y, oof[column].to_numpy(dtype=float))
        ax.plot(fpr, tpr, lw=2 if name == STACK else 1,
                alpha=1.0 if name == STACK else 0.6, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC on held-out stars (accepted candidates)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def probability_histogram(oof: pd.DataFrame, mean_threshold: float, path: Path) -> Path:
    """Score separation, with the average of the per-fold thresholds marked."""
    y = oof["y_true"].to_numpy(dtype=int)
    p = oof[f"prob_{STACK}"].to_numpy(dtype=float)
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, label="FALSE-POSITIVE", color="#b0413e")
    ax.hist(p[y == 1], bins=bins, alpha=0.65, label="CONFIRMED", color="#2e6f9e")
    ax.axvline(
        mean_threshold,
        color="k",
        ls="--",
        lw=1.5,
        label=f"mean fold threshold = {mean_threshold:.4f}",
    )
    ax.set_xlabel("cross-fitted potential_probability")
    ax.set_ylabel("accepted candidates")
    ax.set_title("Score separation on held-out stars")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def importance_plot(summary: pd.DataFrame, path: Path, top_n: int = 20) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.3 * min(top_n, len(summary)) + 1.5)))
    if summary.empty:
        ax.text(0.5, 0.5, "no permutation importance available", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    top = summary.head(top_n).iloc[::-1]
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_fold_mean_std"].fillna(0.0),
        color="#2e6f9e",
        alpha=0.85,
        capsize=2,
    )
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("drop in accepted-candidate average precision")
    ax.set_title(f"Permutation importance on held-out stars (top {min(top_n, len(top))})")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)


def model_comparison_plot(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    order = table.iloc[::-1]
    positions = np.arange(len(order))
    height = 0.38
    ax.barh(positions + height / 2, order["recall"], height=height, label="recall",
            color="#2e6f9e", alpha=0.9)
    ax.barh(positions - height / 2, order["precision"], height=height, label="precision",
            color="#c98a2b", alpha=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{m}{' *' if p else ''}" for m, p in zip(order["model"], order["is_production"])]
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("score at each model's own recall-floor threshold")
    ax.set_title("Pooled held-out performance (* = production stack)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)


def write_all(result: Any, directory: Path) -> list[Path]:
    """Render the full figure set for an evaluation result."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        calibration_plot(result.calibration, directory / "calibration.png"),
        pr_curve_plot(result.oof_predictions, directory / "precision_recall.png"),
        roc_curve_plot(result.oof_predictions, directory / "roc.png"),
        probability_histogram(
            result.oof_predictions,
            float(np.mean(result.fold_thresholds[STACK])),
            directory / "score_separation.png",
        ),
        importance_plot(result.importance_summary(), directory / "permutation_importance.png"),
        model_comparison_plot(result.comparison_table(), directory / "model_comparison.png"),
    ]
    return paths
