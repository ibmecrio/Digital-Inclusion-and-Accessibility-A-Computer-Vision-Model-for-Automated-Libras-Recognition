"""Pytest sanity/regression tests for the Libras Vision pipeline.

These tests guard the dataset integrity, the feature-extraction invariances that
the project relies on, and a minimum performance bar for the shipped models.
They are fast (single fit per model on a small problem) and deterministic.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.data import load_dataset, model_zoo  # noqa: E402
import extract  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_dataset()


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #
def test_feature_dimensionality(data):
    assert data.n_features == 63, "models consume 63 landmark features"
    assert data.X_test.shape[1] == 63


def test_class_count_and_labels(data):
    assert len(data.classes) == 20
    # Dynamic letters must be absent (out of scope).
    for dynamic in ("H", "J", "K", "X", "Z"):
        assert dynamic not in data.classes


def test_test_set_is_balanced(data):
    _, counts = np.unique(data.y_test, return_counts=True)
    assert set(counts.tolist()) == {50}, "held-out test should be 50/class"


def test_no_nans(data):
    assert not np.isnan(data.X_train).any()
    assert not np.isnan(data.X_test).any()


def test_train_test_disjoint_classes_match(data):
    assert set(np.unique(data.y_train)) == set(np.unique(data.y_test))


# --------------------------------------------------------------------------- #
# Feature-extraction invariances (the core normalization contract)
# --------------------------------------------------------------------------- #
def _fake_detection(coords, handedness="Right"):
    """Build a minimal object mimicking a MediaPipe detection result."""
    landmarks = [SimpleNamespace(x=c[0], y=c[1], z=c[2]) for c in coords]
    cat = SimpleNamespace(category_name=handedness)
    return SimpleNamespace(hand_landmarks=[landmarks], handedness=[[cat]])


def _random_hand(seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1, 1, size=(21, 3))
    coords[0] = [0.3, 0.4, 0.0]  # wrist somewhere off-origin
    return coords


def test_translation_invariance_xy():
    """Shifting the hand in the image plane (x, y) must not change features.

    Note: only x and y are made wrist-relative in extract.py; z is kept as
    MediaPipe returns it (already a wrist-relative depth), so the invariance is
    by design limited to the x/y plane.
    """
    coords = _random_hand(1)
    f1, _ = extract.extract_relative_coords(_fake_detection(coords), 0, None)
    shifted = coords + np.array([0.2, -0.1, 0.0])
    f2, _ = extract.extract_relative_coords(_fake_detection(shifted), 0, None)
    assert np.allclose(f1[:63], f2[:63], atol=1e-9)


def test_scale_invariance():
    """Uniformly scaling the hand must not change the extracted features."""
    coords = _random_hand(2)
    f1, _ = extract.extract_relative_coords(_fake_detection(coords), 0, None)
    scaled = coords * 2.5
    f2, _ = extract.extract_relative_coords(_fake_detection(scaled), 0, None)
    assert np.allclose(f1[:63], f2[:63], atol=1e-9)


def test_left_hand_mirroring():
    """A left hand must be mirrored to look like a right hand (x flips)."""
    coords = _random_hand(3)
    right, _ = extract.extract_relative_coords(
        _fake_detection(coords, "Right"), 0, None)
    left, _ = extract.extract_relative_coords(
        _fake_detection(coords, "Left"), 0, None)
    xs = np.arange(0, 63, 3)  # x components
    assert np.allclose(right[xs], -left[xs], atol=1e-9)
    ys = np.arange(1, 63, 3)
    assert np.allclose(right[ys], left[ys], atol=1e-9)


def test_empty_detection_returns_empty():
    empty = SimpleNamespace(hand_landmarks=[], handedness=[])
    feats, wrist = extract.extract_relative_coords(empty, 0, None)
    assert feats.size == 0 and wrist is None


# --------------------------------------------------------------------------- #
# Minimum performance bar (regression guard).
# Each shipped model must clear a conservative held-out accuracy threshold.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", model_zoo(), ids=lambda s: s.key)
def test_model_meets_minimum_accuracy(data, spec):
    model = spec.factory()
    model.fit(data.X_train, data.y_train)
    acc = model.score(data.X_test, data.y_test)
    assert acc >= 0.80, f"{spec.label} held-out accuracy {acc:.3f} below 0.80 floor"
