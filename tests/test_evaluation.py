"""Tests de la couture d'évaluation (`training.evaluation`) — hors-ligne, faux classifieurs.

Cœur du critère d'acceptation (ex-S2) : filtrer les **lignes** (couture) ne change pas les valeurs
de features, contrairement au filtre amont `categories=` qui ampute l'historique avant le calcul.
"""
from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier

from training.backtest import make_backtest_split
from training.config import FeatureConfig
from training.evaluation import (
    CLASSES,
    build_eval_frame,
    predict_proba_frame,
    score,
    select_rows,
)
from training.model import FEATURE_COLUMNS, evaluate, train_classifier


# --- critère d'acceptation : filtre de lignes ≠ filtre d'historique -------------

def test_row_filter_preserves_features_unlike_category_filter(make_res):
    # Un amical (minor) précède un mondial (world) de la même équipe. Le mondial « voit » l'amical
    # dans son historique (home_played == 1) tant qu'on filtre les LIGNES, mais pas si l'amical est
    # retiré du dataset AVANT le calcul des features (filtre `categories=`).
    res = make_res([
        (date(2018, 1, 1), "France", "A", 5, 0),  # amical
        (date(2020, 1, 1), "France", "B", 1, 0),  # mondial
    ]).with_columns(
        tournament_category_label=pl.when(pl.col("match_id") == 0)
        .then(pl.lit("minor"))
        .otherwise(pl.lit("world")),
    )

    # Couture : features sur tout, puis filtre des lignes (le mondial = jeu de test).
    wide = build_eval_frame(FeatureConfig(), res)
    _, test = select_rows(
        wide,
        train_pred=pl.col("date") < date(2019, 1, 1),
        test_pred=(pl.col("date") >= date(2019, 1, 1)) & pl.col("outcome").is_not_null(),
    )
    assert test.get_column("match_id").to_list() == [1]
    assert test.row(0, named=True)["home_played"] == 1  # l'amical a alimenté l'historique

    # Filtre amont `categories=["world"]` : l'amical disparaît AVANT les features -> played == 0.
    filtered = make_backtest_split(
        date(2019, 1, 1), train_years=50, categories=["world"], res=res
    ).test
    assert filtered.row(0, named=True)["home_played"] == 0

    # Donc filtrer les lignes ≠ filtrer l'historique : la couture préserve la feature, pas `categories=`.
    assert test.row(0, named=True)["home_played"] != filtered.row(0, named=True)["home_played"]


def test_build_eval_frame_matches_split_without_category_filter(make_res):
    # Sans filtre de catégorie, build_eval_frame + select_rows reproduit le test de make_backtest_split.
    res = make_res([
        (date(2016, 1, 1), "A", "B", 2, 0),
        (date(2020, 1, 1), "A", "B", 1, 0),
    ])
    wide = build_eval_frame(FeatureConfig(), res)
    _, test = select_rows(
        wide,
        train_pred=pl.col("date") < date(2019, 1, 1),
        test_pred=(pl.col("date") >= date(2019, 1, 1)) & pl.col("outcome").is_not_null(),
    )
    split = make_backtest_split(date(2019, 1, 1), train_years=10, res=res)
    assert test.get_column("match_id").to_list() == split.test.get_column("match_id").to_list()
    assert test.get_column("home_elo").to_list() == split.test.get_column("home_elo").to_list()


# --- predict_proba_frame : probas valides + schéma ------------------------------

def test_predict_proba_frame_is_valid_and_aligned(make_res):
    res = make_res([
        (date(2016, 1, 1), "A", "B", 2, 0),
        (date(2020, 1, 1), "A", "B", 1, 0),
    ])
    split = make_backtest_split(date(2019, 1, 1), train_years=10, res=res)

    class _Fake:
        classes_ = np.array(["away_win", "draw", "home_win"])

        def predict_proba(self, X):
            return np.tile([0.2, 0.3, 0.5], (len(X), 1))

    frame = predict_proba_frame(_Fake(), split.test, FEATURE_COLUMNS)
    for col in ("date", "home_team", "away_team", "outcome", "p_home_win", "p_draw", "p_away_win"):
        assert col in frame.columns
    proba = frame.select("p_away_win", "p_draw", "p_home_win").to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(proba > 0.0) and np.all(proba < 1.0)
    # alignement : la colonne p_home_win porte bien la proba de la classe home_win (0.5).
    np.testing.assert_allclose(frame.get_column("p_home_win").to_numpy(), 0.5, atol=1e-6)


def test_predict_proba_frame_realigns_shuffled_classes(make_res):
    # Si clf.classes_ est dans un autre ordre, les colonnes p_* restent correctement étiquetées.
    res = make_res([
        (date(2016, 1, 1), "A", "B", 2, 0),
        (date(2020, 1, 1), "A", "B", 1, 0),
    ])
    split = make_backtest_split(date(2019, 1, 1), train_years=10, res=res)

    class _Shuffled:
        classes_ = np.array(["home_win", "away_win", "draw"])  # ordre non trié

        def predict_proba(self, X):
            # colonnes dans l'ordre de classes_ : home=0.5, away=0.2, draw=0.3
            return np.tile([0.5, 0.2, 0.3], (len(X), 1))

    frame = predict_proba_frame(_Shuffled(), split.test, FEATURE_COLUMNS)
    np.testing.assert_allclose(frame.get_column("p_home_win").to_numpy(), 0.5, atol=1e-6)
    np.testing.assert_allclose(frame.get_column("p_away_win").to_numpy(), 0.2, atol=1e-6)
    np.testing.assert_allclose(frame.get_column("p_draw").to_numpy(), 0.3, atol=1e-6)


# --- score : reproduit model.evaluate -------------------------------------------

def test_score_reproduces_model_evaluate(make_res):
    res = make_res([
        (date(2016, 1, 1), "A", "B", 2, 0),  # home_win
        (date(2016, 2, 1), "A", "C", 3, 0),  # home_win  -> prior home majoritaire
        (date(2016, 3, 1), "D", "E", 0, 2),  # away_win
        (date(2016, 4, 1), "F", "G", 1, 1),  # draw
        (date(2020, 1, 1), "A", "B", 2, 0),  # home_win (test)
    ])
    split = make_backtest_split(date(2019, 1, 1), train_years=10, res=res)
    clf = train_classifier(split.train, DummyClassifier(strategy="prior"))

    metrics_model = evaluate(clf, split.test, FEATURE_COLUMNS)
    metrics_score = score(predict_proba_frame(clf, split.test, FEATURE_COLUMNS))

    assert metrics_score["log_loss"] == pytest.approx(metrics_model["log_loss"])
    assert metrics_score["accuracy"] == pytest.approx(metrics_model["accuracy"])
    assert metrics_score["n_test"] == metrics_model["n_test"]


def test_classes_order_is_canonical():
    # L'ordre canonique doit être celui de np.unique (alphabétique) — invariant utilisé partout.
    assert CLASSES == ("away_win", "draw", "home_win")
    assert list(CLASSES) == sorted(CLASSES)
