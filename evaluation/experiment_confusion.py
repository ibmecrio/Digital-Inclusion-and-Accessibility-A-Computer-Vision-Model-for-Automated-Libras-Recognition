"""Experiment 3 -- confusion matrices and per-class behaviour on the held-out set.

For every shipped model we fit on the full training set, predict the held-out
test set, render a row-normalized confusion matrix as a PNG, and surface the
most-confused letter pairs plus per-class precision/recall/F1. Row-normalization
(each true class sums to 1) makes the matrices comparable despite per-class
counts and highlights where a class's mass leaks to another letter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend; we only save files
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)

from .data import Dataset, model_zoo

FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _plot_cm(cm_norm: np.ndarray, classes: List[str], title: str,
             out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                  display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
    ax.set_title(title)
    # Thin the in-cell text so a 20x20 grid stays readable.
    for txt in ax.texts:
        txt.set_fontsize(6)
        if txt.get_text() in ("0.00", "0"):
            txt.set_text("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(data: Dataset) -> Dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    classes = data.classes
    out: Dict[str, Dict] = {}

    for spec in model_zoo():
        model = spec.factory()
        model.fit(data.X_train, data.y_train)
        y_pred = model.predict(data.X_test)

        cm = confusion_matrix(data.y_test, y_pred, labels=classes)
        with np.errstate(all="ignore"):
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        fig_path = FIGURES_DIR / f"confusion_{spec.key}.png"
        _plot_cm(cm_norm, classes, f"Matriz de confusao - {spec.label}",
                 fig_path)

        # Most-confused off-diagonal pairs (by absolute misclassification count).
        confusions = []
        for i, true_c in enumerate(classes):
            for j, pred_c in enumerate(classes):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        "true": true_c, "pred": pred_c,
                        "count": int(cm[i, j]),
                        "rate": float(cm_norm[i, j]),
                    })
        confusions.sort(key=lambda d: d["count"], reverse=True)

        prec, rec, f1, support = precision_recall_fscore_support(
            data.y_test, y_pred, labels=classes, zero_division=0)
        per_class = [
            {"class": c, "precision": float(prec[i]), "recall": float(rec[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        ]
        worst = sorted(per_class, key=lambda d: d["f1"])[:5]

        out[spec.label] = {
            "key": spec.key,
            "figure": str(fig_path.relative_to(FIGURES_DIR.parent.parent)),
            "top_confusions": confusions[:8],
            "per_class": per_class,
            "worst_classes": worst,
            "diagonal_min": float(np.min(np.diag(cm_norm))),
        }

    return out
