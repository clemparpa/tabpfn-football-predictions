"""Tests du module de backtesting (découpage train/test sans fuite).

Unitaires sur fixtures synthétiques (valeurs exactes) + un invariant d'intégration sur les
vraies données.
"""
from datetime import date

import polars as pl
import pytest

from training.backtest import add_outcome, make_backtest_split


# --- add_outcome ----------------------------------------------------------------

def test_add_outcome_labels(make_res):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 2, 0),       # home_win
        (date(2020, 1, 2), "C", "D", 0, 3),       # away_win
        (date(2020, 1, 3), "E", "F", 1, 1),       # draw
        (date(2020, 1, 4), "G", "H", None, None),  # non joué -> pas de label
    ])
    out = add_outcome(res).sort("match_id")

    assert out.get_column("match_id").to_list() == [0, 1, 2]  # le non joué est absent
    assert out.get_column("outcome").to_list() == ["home_win", "away_win", "draw"]


# --- fenêtre d'entraînement -----------------------------------------------------

def test_train_window_respects_years(make_res):
    res = make_res([
        (date(2010, 1, 1), "France", "X", 1, 0),  # avant train_start -> exclu
        (date(2018, 1, 1), "France", "Y", 1, 0),  # dans la fenêtre -> train
        (date(2022, 1, 1), "France", "Z", 1, 0),  # >= cutoff -> test
    ])
    split = make_backtest_split(
        date(2021, 1, 1), train_years=5, res=res
    )  # train_start = 2016-01-01

    assert split.train_start == date(2016, 1, 1)
    assert split.train.get_column("match_id").to_list() == [1]
    assert split.train.get_column("outcome").null_count() == 0
    # toutes les dates de train dans [train_start, cutoff)
    assert (split.train.get_column("date") >= split.train_start).all()
    assert (split.train.get_column("date") < split.cutoff).all()


# --- jeu de test ----------------------------------------------------------------

def test_test_set_is_played_from_cutoff(make_res):
    res = make_res([
        (date(2020, 1, 1), "France", "A", 1, 0),       # train
        (date(2022, 1, 1), "France", "B", 2, 1),       # >= cutoff, joué -> test
        (date(2023, 1, 1), "France", "C", None, None),  # >= cutoff, jamais joué -> exclu
    ])
    split = make_backtest_split(date(2021, 1, 1), train_years=10, res=res)

    assert split.test.get_column("match_id").to_list() == [1]  # le fixture sans score est exclu
    assert (split.test.get_column("date") >= split.cutoff).all()
    assert split.test.get_column("outcome").null_count() == 0


def test_test_features_are_populated(make_res):
    # un match du test reçoit ses features de forme pré-match, jamais null
    res = make_res([
        (date(2020, 1, 1), "France", "A", 2, 0),
        (date(2022, 1, 1), "France", "B", 1, 0),
    ])
    split = make_backtest_split(date(2021, 1, 1), train_years=10, res=res)
    test_row = split.test.row(0, named=True)
    assert test_row["home_points_history"] is not None
    assert test_row["home_played"] == 1  # 1 match joué avant ce match


# --- rolling : la forme reflète les matchs intra-fenêtre (sans self-leak) -------

def test_rolling_form_reflects_intra_window(make_res):
    # France gagne 2 fois avant le cutoff, PERD en 2022, puis rejoue en 2023.
    # 2022 et 2023 sont tous deux dans le test : le match de 2023 DOIT voir le résultat de
    # 2022 (rolling, comme la baseline), mais le match de 2022 ne doit PAS voir son propre
    # résultat (pré-match strict).
    res = make_res([
        (date(2020, 1, 1), "France", "A", 2, 0),  # victoire (3 pts)
        (date(2020, 6, 1), "France", "B", 2, 0),  # victoire (3 pts)
        (date(2022, 1, 1), "France", "C", 0, 1),  # défaite (0 pt) — match du test
        (date(2023, 1, 1), "France", "D", 3, 0),  # match du test
    ])
    split = make_backtest_split(date(2021, 1, 1), train_years=10, res=res)
    test = split.test.sort("match_id")

    assert test.get_column("match_id").to_list() == [2, 3]
    # played : 2022 voit 2 matchs avant lui ; 2023 en voit 3 (le match de 2022 a compté)
    assert test.get_column("home_played").to_list() == [2, 3]
    # 2022 : pré-match strict = mean([3, 3]) = 3.0 (sa propre défaite n'y est pas)
    # 2023 : la défaite de 2022 entre dans l'historique = mean([3, 3, 0]) = 2.0
    assert test.get_column("home_points_history").to_list() == [3.0, 2.0]


# --- filtre par catégorie : retrait complet du dataset --------------------------

def test_category_filter_removes_from_history(make_res):
    res = make_res([
        (date(2018, 1, 1), "France", "A", 5, 0),  # amical (minor)
        (date(2020, 1, 1), "France", "B", 1, 0),  # mondial (world)
    ])
    res = res.with_columns(
        tournament_category=pl.when(pl.col("match_id") == 0).then(5).otherwise(1),
        tournament_category_label=pl.when(pl.col("match_id") == 0)
        .then(pl.lit("minor"))
        .otherwise(pl.lit("world")),
    )
    split = make_backtest_split(
        date(2030, 1, 1), train_years=50, categories=["world"], res=res
    )

    # l'amical a disparu (du dataset entier) ...
    assert split.train.get_column("match_id").to_list() == [1]
    # ... et n'a pas alimenté l'historique : le mondial a played == 0
    assert split.train.row(0, named=True)["home_played"] == 0


def test_unknown_category_raises(make_res):
    res = make_res([(date(2020, 1, 1), "A", "B", 1, 0)])
    with pytest.raises(ValueError, match="inconnue"):
        make_backtest_split(date(2021, 1, 1), train_years=5, categories=["bogus"], res=res)


# --- intégration : cohérence et anti-fuite sur vraies données -------------------

def test_real_split_is_coherent(real_matches):
    cutoff = date(2018, 1, 1)
    split = make_backtest_split(cutoff, train_years=4, res=real_matches)

    assert split.train.height > 0
    assert split.test.height > 0
    assert split.train_start == date(2014, 1, 1)

    # bornes de dates
    assert (split.train.get_column("date") >= split.train_start).all()
    assert (split.train.get_column("date") < cutoff).all()
    assert (split.test.get_column("date") >= cutoff).all()

    # features de forme du test : aucune null
    for side in ("home", "away"):
        assert split.test.get_column(f"{side}_points_history").null_count() == 0
        assert split.test.get_column(f"{side}_played").null_count() == 0

    # invariant anti-fuite (rolling) : `played` d'un match du test = nb de matchs JOUÉS de
    # l'équipe strictement AVANT ce match — ordre (date, match_id) — recalculé depuis le brut.
    finished = real_matches.filter(pl.col("finished"))
    appearances = pl.concat([
        finished.select(pl.col("home_team").alias("team"),
                        pl.col("date").alias("adate"), pl.col("match_id").alias("amid")),
        finished.select(pl.col("away_team").alias("team"),
                        pl.col("date").alias("adate"), pl.col("match_id").alias("amid")),
    ])

    for side in ("home", "away"):
        check = (
            split.test.select(
                "match_id",
                observed=pl.col(f"{side}_played"),
                team=pl.col(f"{side}_team"),
                mdate=pl.col("date"),
            )
            .join(appearances, on="team", how="left")
            .with_columns(
                prior=(
                    (pl.col("adate") < pl.col("mdate"))
                    | ((pl.col("adate") == pl.col("mdate")) & (pl.col("amid") < pl.col("match_id")))
                ).fill_null(False).cast(pl.Int32)
            )
            .group_by("match_id", "observed")
            .agg(recount=pl.col("prior").sum())
        )
        assert check.filter(pl.col("observed") != pl.col("recount")).height == 0
