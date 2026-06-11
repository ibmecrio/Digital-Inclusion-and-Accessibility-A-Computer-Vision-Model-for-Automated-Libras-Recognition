"""Statistical helpers used to give the comparison scientific footing.

We use:
  * Friedman test  -- omnibus test for "do the models differ at all?" across
    the paired cross-validation folds (non-parametric, repeated-measures).
  * Wilcoxon signed-rank with Holm correction -- pairwise post-hoc on the same
    paired folds.
  * McNemar's test -- pairwise comparison on the single held-out test set,
    which is the statistically correct test when two classifiers are compared
    on the *same* set of test items.
  * Bootstrap / normal CIs for point estimates.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, half-width, std) using a t-based CI for small samples."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0, 0.0
    sem = stats.sem(values)
    half = float(sem * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, half, float(values.std(ddof=1))


def friedman(fold_scores: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Friedman test across models. fold_scores maps model -> per-fold scores
    (all arrays must be aligned to the same folds)."""
    keys = list(fold_scores.keys())
    matrix = [np.asarray(fold_scores[k], dtype=float) for k in keys]
    stat, p = stats.friedmanchisquare(*matrix)
    # Average ranks (higher score -> better -> rank 1). Used for reporting.
    arr = np.vstack(matrix).T  # folds x models
    # rank within each fold, higher is better => negate before ranking
    ranks = np.vstack([stats.rankdata(-row) for row in arr])
    avg_ranks = {k: float(ranks[:, i].mean()) for i, k in enumerate(keys)}
    return {"statistic": float(stat), "p_value": float(p), "avg_ranks": avg_ranks}


def pairwise_wilcoxon_holm(fold_scores: Dict[str, np.ndarray]) -> List[Dict]:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction."""
    keys = list(fold_scores.keys())
    pairs = list(combinations(keys, 2))
    raw = []
    for a, b in pairs:
        sa = np.asarray(fold_scores[a], dtype=float)
        sb = np.asarray(fold_scores[b], dtype=float)
        if np.allclose(sa, sb):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(sa, sb)
            except ValueError:
                p = 1.0
        raw.append([a, b, float(p), float(sa.mean() - sb.mean())])

    # Holm correction over the m p-values.
    m = len(raw)
    order = sorted(range(m), key=lambda i: raw[i][2])
    holm = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * raw[idx][2])
        running_max = max(running_max, adj)
        holm[idx] = running_max
    results = []
    for i, (a, b, p, diff) in enumerate(raw):
        results.append({
            "a": a, "b": b, "p_raw": p, "p_holm": holm[i],
            "mean_diff": diff, "significant": holm[i] < 0.05,
        })
    return results


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    """McNemar's test comparing two classifiers on the same test set.

    Uses the exact binomial test on the discordant pairs when their count is
    small, otherwise the chi-square approximation with continuity correction.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    # b: A right, B wrong; c: A wrong, B right
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if n < 25:
        # Exact two-sided binomial test, p=0.5 under H0.
        p = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": p,
                "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(stats.chi2.sf(stat, 1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p,
            "method": "chi2"}
