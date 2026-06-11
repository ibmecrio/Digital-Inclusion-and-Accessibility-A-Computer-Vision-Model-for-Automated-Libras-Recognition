"""Experiment 2 -- sweep alternative hyperparameter configurations per family.

For every model family we cross-validate a grid of configurations on the
training set (StratifiedKFold, shared across configs of the same family) and
rank them by mean accuracy with a 95% CI. The configuration that the repo ships
is flagged so the report can say whether the shipped choice is defensible.

The held-out test set is deliberately *not* used here -- using it to pick
hyperparameters would leak the test set and inflate the final numbers reported
in Experiment 1.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import stats
from .data import RANDOM_STATE, Dataset, config_sweeps

N_SPLITS = 5


def run(data: Dataset) -> Dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    results: Dict[str, Dict] = {}

    for sweep in config_sweeps():
        rows: List[Dict] = []
        for desc, factory in sweep.configs:
            t0 = time.perf_counter()
            scores = cross_val_score(factory(), data.X_train, data.y_train,
                                     cv=cv, scoring="accuracy", n_jobs=-1)
            elapsed = time.perf_counter() - t0
            mean, half, std = stats.mean_ci(scores)
            rows.append({
                "config": desc,
                "acc_mean": mean,
                "acc_ci95": half,
                "acc_std": std,
                "seconds": elapsed,
                "is_repo_default": desc == sweep.repo_default,
            })
        rows.sort(key=lambda r: r["acc_mean"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

        repo_row = next((r for r in rows if r["is_repo_default"]), None)
        best_row = rows[0]
        results[sweep.family] = {
            "n_splits": N_SPLITS,
            "rows": rows,
            "best": best_row,
            "repo_default": repo_row,
        }

    return results
