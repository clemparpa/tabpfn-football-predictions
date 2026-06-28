"""Tests du chargement des données sur les vraies données (results.csv) — invariants."""
import polars as pl

from training.data import load_tournament_importance


def test_tournament_importance_shape():
    imp = load_tournament_importance()
    assert "tournament_category" in imp.columns
    assert "tournament_category_label" in imp.columns
    assert "category" not in imp.columns  # remplacée
    assert imp.get_column("tournament_category").is_between(1, 7).all()


def test_load_matches_columns(real_matches):
    expected = {
        "match_id", "date", "home_team", "away_team",
        "home_score", "away_score", "neutral", "finished",
        "tournament_category", "tournament_category_label",
    }
    assert expected.issubset(set(real_matches.columns))
    assert "tournament" in real_matches.columns  # conservée (cible du backtest tournoi)
    assert real_matches.schema["date"] == pl.Date


def test_match_id_unique(real_matches):
    assert real_matches.get_column("match_id").n_unique() == real_matches.height


def test_finished_flag_matches_scores(real_matches):
    recomputed = (
        real_matches.get_column("home_score").is_not_null()
        & real_matches.get_column("away_score").is_not_null()
    )
    assert recomputed.equals(real_matches.get_column("finished"))
