"""Tests du modèle d'ensemble (hors-ligne : faux TabPFN injecté, aucun appel réseau).

On vérifie les maths pures de `combine_probas`, le comportement de bout en bout de
`EnsembleClassifier` (avec un `DummyClassifier` à la place de TabPFN), et la reconstruction
de l'`EnsembleConfig` depuis des params MLflow.
"""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from training.ensemble import (
    EnsembleClassifier,
    EnsembleConfig,
    build_model,
    combine_probas,
)
from training.mlflow_io import reconstruct_ensemble_config


# --- combine_probas -------------------------------------------------------------

def test_combine_weight_one_returns_tabpfn():
    p_a = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    p_b = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    out = combine_probas(p_a, p_b, method="arith", weight=1.0)
    np.testing.assert_allclose(out, p_a, atol=1e-6)


def test_combine_weight_zero_returns_gbm():
    p_a = np.array([[0.7, 0.2, 0.1]])
    p_b = np.array([[0.1, 0.3, 0.6]])
    out = combine_probas(p_a, p_b, method="arith", weight=0.0)
    np.testing.assert_allclose(out, p_b, atol=1e-6)


def test_combine_arith_is_weighted_mean():
    p_a = np.array([[0.6, 0.3, 0.1]])
    p_b = np.array([[0.2, 0.2, 0.6]])
    out = combine_probas(p_a, p_b, method="arith", weight=0.5)
    expected = 0.5 * p_a + 0.5 * p_b  # déjà normalisé (somme 1)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_combine_geom_matches_normalized_geometric_mean():
    p_a = np.array([[0.6, 0.3, 0.1]])
    p_b = np.array([[0.2, 0.2, 0.6]])
    out = combine_probas(p_a, p_b, method="geom", weight=0.5)
    raw = np.sqrt(p_a * p_b)
    expected = raw / raw.sum()
    np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("method", ["arith", "geom"])
def test_combine_output_is_valid_distribution(method):
    rng = np.random.default_rng(0)
    p_a = rng.dirichlet([1, 1, 1], size=5)
    p_b = rng.dirichlet([1, 1, 1], size=5)
    out = combine_probas(p_a, p_b, method=method, weight=0.4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_combine_unknown_method_raises():
    p = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError):
        combine_probas(p, p, method="harmonic", weight=0.5)


# --- EnsembleClassifier ---------------------------------------------------------

_X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
_Y = np.array(["away_win", "draw", "home_win", "home_win", "draw", "away_win"])


def _fake_tabpfn():
    return DummyClassifier(strategy="prior")


def test_tabpfn_only_matches_member_proba():
    cfg = EnsembleConfig(use_gbm=False)
    clf = EnsembleClassifier(cfg, tabpfn=_fake_tabpfn()).fit(_X, _Y)
    member = _fake_tabpfn().fit(_X, _Y)
    # classes_ alignées sur np.unique(y) ; le DummyClassifier 'prior' renvoie la même proba partout.
    expected = member.predict_proba(_X)
    np.testing.assert_allclose(clf.predict_proba(_X), expected, atol=1e-6)
    np.testing.assert_array_equal(clf.classes_, np.unique(_Y))


def test_ensemble_with_gbm_differs_and_is_valid():
    cfg = EnsembleConfig(
        use_gbm=True, combine="arith", weight=0.5,
        gbm_kwargs={"max_iter": 20, "max_depth": 2},
    )
    clf = EnsembleClassifier(cfg, tabpfn=_fake_tabpfn()).fit(_X, _Y)
    proba = clf.predict_proba(_X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(proba > 0.0) and np.all(proba < 1.0)
    # predict cohérent avec l'argmax des probas combinées.
    np.testing.assert_array_equal(clf.predict(_X), clf.classes_[proba.argmax(axis=1)])


def test_predict_proba_before_fit_raises():
    clf = build_model(EnsembleConfig(), random_state=0)
    with pytest.raises(RuntimeError):
        clf.predict_proba(_X)


def test_build_model_returns_ensemble_classifier():
    clf = build_model(EnsembleConfig(tabpfn_kwargs={"n_estimators": 2}), random_state=7)
    assert isinstance(clf, EnsembleClassifier)
    assert clf.cfg.tabpfn_kwargs == {"n_estimators": 2}
    assert clf.random_state == 7


# --- reconstruct_ensemble_config ------------------------------------------------

def test_reconstruct_ensemble_config_reads_ensemble_block():
    params = {
        "ensemble.use_gbm": "True",
        "ensemble.combine": "geom",
        "ensemble.weight": "0.65",
        "gbm.learning_rate": "0.05",
        "gbm.max_iter": "200",
        "gbm.max_depth": "3",
        "gbm.l2_regularization": "1.5",
    }
    cfg = reconstruct_ensemble_config(params, {"n_estimators": 8})
    assert cfg.use_gbm is True
    assert cfg.combine == "geom"
    assert cfg.weight == 0.65
    assert cfg.gbm_kwargs == {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": 3,
        "l2_regularization": 1.5,
    }
    assert cfg.tabpfn_kwargs == {"n_estimators": 8}


def test_reconstruct_ensemble_config_defaults_to_tabpfn_only():
    cfg = reconstruct_ensemble_config({}, {"n_estimators": 2})
    assert cfg.use_gbm is False
    assert cfg.weight == 1.0
    assert cfg.gbm_kwargs == {}
    assert cfg.tabpfn_kwargs == {"n_estimators": 2}
