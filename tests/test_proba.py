"""Tests du post-traitement des probas (`training.proba`) — maths pures, hors-ligne."""
import numpy as np
import pytest

from training.proba import PROBA_EPS, align_proba, clip_renorm


# --- clip_renorm ----------------------------------------------------------------

def test_clip_renorm_rows_sum_to_one():
    proba = np.array([[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    out = clip_renorm(proba)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_clip_renorm_bounds_strictly_inside_open_interval():
    # une proba à 0 et une à 1 : après clip+renorm, tout est strictement dans (0, 1).
    proba = np.array([[1.0, 0.0, 0.0]])
    out = clip_renorm(proba)
    assert np.all(out > 0.0) and np.all(out < 1.0)
    # la masse écrasée reste minuscule (de l'ordre de eps), pas redistribuée arbitrairement.
    assert out[0, 1] == pytest.approx(PROBA_EPS, rel=1e-3)


def test_clip_renorm_leaves_valid_proba_almost_unchanged():
    proba = np.array([[0.1, 0.2, 0.7]])
    np.testing.assert_allclose(clip_renorm(proba), proba, atol=1e-6)


def test_clip_renorm_renormalizes_unnormalized_input():
    # entrée sommant à 0.9994 (dérive float32 de TabPFN) -> renormalisée à 1.
    proba = np.array([[0.1, 0.2, 0.7]]) * 0.9994
    np.testing.assert_allclose(clip_renorm(proba), [[0.1, 0.2, 0.7]], atol=1e-6)


# --- align_proba ----------------------------------------------------------------

def test_align_proba_reorders_columns_to_target():
    # colonnes étiquetées [away_win, draw, home_win] -> cible [home_win, draw, away_win].
    proba = np.array([[0.1, 0.2, 0.7]])
    out = align_proba(proba, ["away_win", "draw", "home_win"], ["home_win", "draw", "away_win"])
    np.testing.assert_allclose(out, [[0.7, 0.2, 0.1]], atol=1e-12)


def test_align_proba_identity_when_orders_match():
    proba = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    classes = ["away_win", "draw", "home_win"]
    np.testing.assert_allclose(align_proba(proba, classes, classes), proba, atol=1e-12)


def test_align_proba_raises_on_missing_target_class():
    proba = np.array([[0.4, 0.6]])
    with pytest.raises(ValueError):
        align_proba(proba, ["away_win", "home_win"], ["away_win", "draw", "home_win"])
