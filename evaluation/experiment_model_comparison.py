"""Experiment 1 -- compare the five shipped classifiers with scientific rigor.

Protocol
--------
* Model selection signal: RepeatedStratifiedKFold (5 splits x 3 repeats = 15
  paired estimates) on the *training* set only. Every model is evaluated on the
  identical folds, so the per-fold scores are paired and admit repeated-measures
  statistics.
* Generalization estimate: each model is retrained on the full training set and
  evaluated once on the held-out test set (never seen during CV), reporting
  accuracy plus macro/weighted precision, recall and F1.
* Significance: Friedman omnibus + Wilcoxon/Holm post-hoc on the CV folds, and
  McNemar between the two best models on the held-out test predictions.
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from . import stats
from .data import RANDOM_STATE, Dataset, model_zoo

N_SPLITS = 5
N_REPEATS = 3


def run(data: Dataset) -> Dict:
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_STATE)
    specs = model_zoo()

    cv_scores: Dict[str, np.ndarray] = {}
    per_model = {}

    for spec in specs:
        t0 = time.perf_counter()
        scores = cross_val_score(spec.factory(), data.X_train, data.y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1)
        cv_time = time.perf_counter() - t0
        mean, half, std = stats.mean_ci(scores)
        cv_scores[spec.label] = scores

        # Held-out test evaluation on a fresh fit over the full training set.
        t0 = time.perf_counter()
        model = spec.factory()
        model.fit(data.X_train, data.y_train)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = model.predict(data.X_test)
        predict_time = time.perf_counter() - t0

        per_model[spec.label] = {
            "key": spec.key,
            "repo_config": spec.repo_config,
            "cv_acc_mean": mean,
            "cv_acc_ci95": half,
            "cv_acc_std": std,
            "cv_scores": scores.tolist(),
            "test_accuracy": float(accuracy_score(data.y_test, y_pred)),
            "test_precision_macro": float(precision_score(
                data.y_test, y_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(
                data.y_test, y_pred, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(
                data.y_test, y_pred, average="macro", zero_division=0)),
            "test_f1_weighted": float(f1_score(
                data.y_test, y_pred, average="weighted", zero_division=0)),
            "cv_seconds": cv_time,
            "fit_seconds": fit_time,
            "predict_seconds": predict_time,
            "predict_ms_per_sample": predict_time / len(data.X_test) * 1000.0,
            "y_pred": y_pred.tolist(),
        }

    friedman = stats.friedman(cv_scores)
    posthoc = stats.pairwise_wilcoxon_holm(cv_scores)

    # McNemar between the two best models by held-out accuracy.
    ranked = sorted(per_model.items(), key=lambda kv: kv[1]["test_accuracy"],
                    reverse=True)
    best_label, second_label = ranked[0][0], ranked[1][0]
    mcnemar = stats.mcnemar(
        data.y_test,
        np.array(per_model[best_label]["y_pred"]),
        np.array(per_model[second_label]["y_pred"]),
    )

    return {
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "models": per_model,
        "friedman": friedman,
        "posthoc_wilcoxon_holm": posthoc,
        "mcnemar_top2": {"model_a": best_label, "model_b": second_label,
                         **mcnemar},
        "ranking": [label for label, _ in ranked],
    }
