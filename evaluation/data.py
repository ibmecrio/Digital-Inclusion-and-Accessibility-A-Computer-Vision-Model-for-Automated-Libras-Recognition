"""Data loading and model/configuration definitions shared by every experiment.

The CSVs hold one row per detected hand:
    column 0      -> pandas index written by landmark_extractor.py (dropped)
    columns 1..63 -> the 63 relative landmark coordinates (x0,y0,z0 ... x20,y20,z20)
    last column   -> the letter label

These are exactly the features the production models consume (see
``libras_vision.py``, which slices ``landmarks_arr[:63]``), so the evaluation
operates on the same feature space as the deployed system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Repo root = parent of this package directory.
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "landmarks_training.csv"
TEST_CSV = ROOT / "landmarks_test.csv"

# A single global seed makes every split, sweep and model reproducible.
RANDOM_STATE = 42


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: List[str]
    feature_names: List[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_dataset() -> Dataset:
    """Load the cached training/test landmark CSVs into numpy arrays.

    Labels are kept as strings; estimators below handle the string targets
    directly (sklearn classifiers accept string labels), which keeps confusion
    matrices and reports human-readable without a separate LabelEncoder.
    """
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    feature_cols = [c for c in train.columns if c not in ("Unnamed: 0", "label")]

    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=str)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=str)

    classes = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    return Dataset(X_train, y_train, X_test, y_test, classes, feature_cols)


# --------------------------------------------------------------------------- #
# Model zoo -- "production" configurations.
#
# Each factory reproduces exactly the estimator that the corresponding training
# script in the repo builds, so the comparison reflects what actually ships.
# probability=True is dropped from the SVM here: it is only needed for the
# real-time confidence score and roughly triples fit time without changing the
# hard predictions used for accuracy/confusion analysis.
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    key: str
    label: str
    factory: Callable[[], BaseEstimator]
    repo_config: str  # human-readable description of the shipped configuration


def _knn() -> BaseEstimator:
    return KNeighborsClassifier(n_neighbors=21, weights="uniform")


def _logreg() -> BaseEstimator:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _mlp() -> BaseEstimator:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                      random_state=RANDOM_STATE),
    )


def _rf() -> BaseEstimator:
    return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                  random_state=RANDOM_STATE)


def _svm() -> BaseEstimator:
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale"),
    )


def model_zoo() -> List[ModelSpec]:
    """The five shipped classifiers, in a stable order."""
    return [
        ModelSpec("knn", "KNN", _knn, "k=21, weights=uniform"),
        ModelSpec("logreg", "Regressao Logistica", _logreg,
                  "StandardScaler + LogisticRegression(max_iter=1000)"),
        ModelSpec("svm", "SVM (RBF)", _svm,
                  "StandardScaler + SVC(kernel=rbf, C=10, gamma=scale)"),
        ModelSpec("rf", "Random Forest", _rf,
                  "RandomForestClassifier(n_estimators=300)"),
        ModelSpec("mlp", "MLP", _mlp,
                  "StandardScaler + MLPClassifier(hidden=(128,64), max_iter=500)"),
    ]


# --------------------------------------------------------------------------- #
# Configuration sweeps -- alternative hyperparameters per family.
#
# Each entry is (description, estimator factory). The factories are evaluated by
# cross-validation in experiment_hyperparameters.py. The configuration that
# matches the shipped model is flagged via ``repo_default``.
# --------------------------------------------------------------------------- #
@dataclass
class ConfigSweep:
    family: str
    configs: List[tuple]  # (description, factory)
    repo_default: str     # description string that equals the shipped config


def _scaled(estimator: BaseEstimator) -> BaseEstimator:
    return make_pipeline(StandardScaler(), estimator)


def config_sweeps() -> List[ConfigSweep]:
    knn_configs = []
    for k in (1, 3, 5, 11, 21, 31):
        for w in ("uniform", "distance"):
            knn_configs.append((
                f"k={k}, weights={w}",
                lambda k=k, w=w: KNeighborsClassifier(n_neighbors=k, weights=w),
            ))

    logreg_configs = [
        (f"C={c}", lambda c=c: _scaled(LogisticRegression(C=c, max_iter=1000)))
        for c in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]

    svm_configs = []
    for kernel in ("linear", "rbf", "poly"):
        for c in (0.1, 1.0, 10.0, 100.0):
            svm_configs.append((
                f"kernel={kernel}, C={c}",
                lambda kernel=kernel, c=c: _scaled(
                    SVC(kernel=kernel, C=c, gamma="scale")),
            ))

    rf_configs = []
    for n in (50, 100, 300, 500):
        for depth in (None, 10, 20):
            rf_configs.append((
                f"n_estimators={n}, max_depth={depth}",
                lambda n=n, depth=depth: RandomForestClassifier(
                    n_estimators=n, max_depth=depth, n_jobs=-1,
                    random_state=RANDOM_STATE),
            ))

    mlp_configs = []
    for hidden in ((64,), (128,), (128, 64), (256, 128)):
        for alpha in (1e-4, 1e-2):
            mlp_configs.append((
                f"hidden={hidden}, alpha={alpha}",
                lambda hidden=hidden, alpha=alpha: _scaled(
                    MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                  max_iter=500, random_state=RANDOM_STATE)),
            ))

    return [
        ConfigSweep("KNN", knn_configs, "k=21, weights=uniform"),
        ConfigSweep("Regressao Logistica", logreg_configs, "C=1.0"),
        ConfigSweep("SVM", svm_configs, "kernel=rbf, C=10.0"),
        ConfigSweep("Random Forest", rf_configs, "n_estimators=300, max_depth=None"),
        ConfigSweep("MLP", mlp_configs, "hidden=(128, 64), alpha=0.0001"),
    ]
