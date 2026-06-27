"""Tests du module d'entraînement/évaluation (offline, sans réseau ni MLflow).

On injecte un faux classifieur (sklearn `DummyClassifier` ou un espion) pour valider toute
l'orchestration sans appeler l'API TabPFN. L'intégration features/split est vérifiée sur les
vraies données.
"""
from datetime import date

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from training.backtest import make_backtest_split
from training.model import (
    FEATURE_COLUMNS,
    _feature_matrix,
    evaluate,
    run_backtest,
)

CUTOFF = date(2018, 1, 1)


@pytest.fixture(scope="module")
def real_split(real_matches):
    return make_backtest_split(CUTOFF, train_years=4, res=real_matches)


# --- features -------------------------------------------------------------------

def test_feature_columns_present_and_numeric(real_split):
    test = real_split.test
    for col in FEATURE_COLUMNS:
        assert col in test.columns

    matrix = _feature_matrix(test)
    assert matrix.shape == (test.height, len(FEATURE_COLUMNS))
    assert np.issubdtype(matrix.dtype, np.floating)
    assert not np.isnan(matrix).any()  # report de forme sur le test -> aucun NaN


def test_neutral_is_numeric(make_res):
    res = make_res(
        [
            (date(2016, 1, 1), "A", "B", 1, 0),
            (date(2020, 1, 1), "A", "B", 2, 1),
        ],
        neutral=True,
    )
    split = make_backtest_split(date(2019, 1, 1), train_years=10, res=res)
    idx = FEATURE_COLUMNS.index("neutral")
    column = _feature_matrix(split.test)[:, idx]
    assert set(np.unique(column)).issubset({0.0, 1.0})
    assert column[0] == 1.0  # neutral=True -> 1


# --- orchestration train + évaluation -------------------------------------------

def test_train_and_evaluate_offline(real_split):
    result = run_backtest(
        CUTOFF,
        train_years=4,
        classifier=DummyClassifier(strategy="prior"),
        log_mlflow=False,
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert np.isfinite(result.log_loss)
    assert set(result.classes).issubset({"home_win", "away_win", "draw"})
    assert result.n_train == real_split.train.height
    assert result.n_test == real_split.test.height


def test_evaluate_handles_class_absent_from_test(make_res):
    # Le test ne contient qu'une seule classe (home_win) alors que le modèle en connaît 3.
    # labels=clf.classes_ est indispensable : sinon log_loss déduit les labels de y_true seul
    # et ne peut plus aligner les 3 colonnes de proba.
    res = make_res(
        [
            (date(2016, 1, 1), "A", "B", 2, 0),  # home_win (train)
            (date(2020, 1, 1), "A", "B", 1, 0),  # home_win (test, classe unique)
        ]
    )
    split = make_backtest_split(date(2019, 1, 1), train_years=10, res=res)

    class _ThreeClassSpy:
        classes_ = np.array(["away_win", "draw", "home_win"])  # trié, comme un vrai classifieur

        def predict(self, X):
            return np.array(["home_win"] * len(X))

        def predict_proba(self, X):
            return np.tile([0.1, 0.2, 0.7], (len(X), 1))  # 0.7 sur la colonne home_win

    metrics = evaluate(_ThreeClassSpy(), split.test)
    assert metrics["accuracy"] == 1.0
    assert metrics["log_loss"] == pytest.approx(-np.log(0.7))  # -log(proba vraie classe)
    assert metrics["n_test"] == split.test.height


# --- plafond d'entraînement -----------------------------------------------------

class _RowCountSpy:
    """Faux classifieur qui retient le nombre de lignes vues à fit."""

    classes_ = np.array(["away_win", "draw", "home_win"])

    def fit(self, X, y):
        self.n_rows_seen = len(X)
        return self

    def predict(self, X):
        return np.array(["home_win"] * len(X))

    def predict_proba(self, X):
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(X), 1))


def test_max_train_caps_rows(real_split):
    spy = _RowCountSpy()
    cap = 100
    result = run_backtest(
        CUTOFF,
        train_years=4,
        classifier=spy,
        max_train=cap,
        log_mlflow=False,
    )
    assert real_split.train.height > cap  # le plafond est bien sollicité
    assert spy.n_rows_seen == cap
    assert result.n_train == cap
