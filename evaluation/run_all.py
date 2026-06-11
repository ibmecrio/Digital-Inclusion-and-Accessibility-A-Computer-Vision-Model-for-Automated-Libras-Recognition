"""Orchestrate every experiment and emit the technical report + artifacts.

Usage:
    python -m evaluation.run_all
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import (experiment_confusion, experiment_hyperparameters,
               experiment_model_comparison, report)
from .data import RANDOM_STATE, load_dataset

ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
REPORT_MD = ROOT / "RELATORIO_TECNICO.md"


def _strip_heavy(comparison: dict) -> dict:
    """Drop the per-sample prediction arrays before serializing to JSON."""
    out = json.loads(json.dumps(comparison, default=float))
    for m in out["models"].values():
        m.pop("y_pred", None)
    return out


def main() -> None:
    np.random.seed(RANDOM_STATE)
    t_start = time.perf_counter()

    print("Carregando dataset...")
    data = load_dataset()
    meta = {
        "random_state": RANDOM_STATE,
        "n_train": int(len(data.X_train)),
        "n_test": int(len(data.X_test)),
        "n_classes": int(len(data.classes)),
        "n_features": int(data.n_features),
        "classes": data.classes,
    }
    print(f"  treino={meta['n_train']}  teste={meta['n_test']}  "
          f"classes={meta['n_classes']}  atributos={meta['n_features']}")

    print("Experimento 1: comparacao entre modelos (CV + held-out + estatistica)...")
    comparison = experiment_model_comparison.run(data)

    print("Experimento 2: varredura de hiperparametros...")
    hyper = experiment_hyperparameters.run(data)

    print("Experimento 3: matrizes de confusao...")
    confusion = experiment_confusion.run(data)

    print("Gerando relatorio...")
    md = report.build_report(meta, comparison, hyper, confusion)
    REPORT_MD.write_text(md, encoding="utf-8")

    RESULTS_JSON.write_text(json.dumps({
        "meta": meta,
        "comparison": _strip_heavy(comparison),
        "hyperparameters": json.loads(json.dumps(hyper, default=float)),
        "confusion": json.loads(json.dumps(confusion, default=float)),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print(f"\nConcluido em {elapsed:.1f}s")
    print(f"  Relatorio:  {REPORT_MD}")
    print(f"  Resultados: {RESULTS_JSON}")
    print(f"  Figuras:    {experiment_confusion.FIGURES_DIR}")


if __name__ == "__main__":
    main()
