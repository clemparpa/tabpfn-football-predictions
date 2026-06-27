"""Tests d'intégration du pipeline complet sur les vraies données — invariants.

On vérifie des propriétés robustes (pas de valeurs épinglées) afin que les tests
survivent à un rafraîchissement de results.csv.
"""
import polars as pl
import pytest

from training.features import build_features
from training.features.team_form import _FEATURE_COLUMNS


@pytest.fixture(scope="module")
def wide(real_matches, default_cfg):
    return build_features(real_matches, default_cfg)


def test_row_count_preserved(wide, real_matches):
    # le join left ne doit perdre aucun match
    assert wide.height == real_matches.height


def test_match_id_unique(wide):
    assert wide.get_column("match_id").n_unique() == wide.height


def test_expected_columns_present(wide):
    for col in _FEATURE_COLUMNS:
        assert f"home_{col}" in wide.columns
        assert f"away_{col}" in wide.columns
    assert "points_history_diff" in wide.columns
    assert "goal_diff_history_diff" in wide.columns


def test_future_matches_have_null_features(wide):
    future = wide.filter(~pl.col("finished"))
    if future.height == 0:
        pytest.skip("aucun match non joué dans le jeu de données")
    assert future.get_column("home_points_history").null_count() == future.height


def test_played_matches_have_non_null_features(wide):
    played = wide.filter(pl.col("finished"))
    assert played.get_column("home_points_history").null_count() == 0


def test_diffs_are_consistent(wide):
    played = wide.filter(pl.col("finished"))
    pts = played.select(
        (pl.col("points_history_diff")
         - (pl.col("home_points_history") - pl.col("away_points_history"))).abs().max()
    ).item()
    gd = played.select(
        (pl.col("goal_diff_history_diff")
         - (pl.col("home_goal_diff_history") - pl.col("away_goal_diff_history"))).abs().max()
    ).item()
    assert pts < 1e-9
    assert gd < 1e-9


def test_no_leakage_cold_start(wide, default_cfg):
    # played == 0  <=>  aucun match antérieur  =>  features = defaults
    for side in ("home", "away"):
        cold = wide.filter(pl.col(f"{side}_played") == 0)
        assert cold.height > 0
        assert (cold.get_column(f"{side}_points_history") == default_cfg.default_points).all()
        assert (cold.get_column(f"{side}_win_streak") == 0).all()
        assert (cold.get_column(f"{side}_draw_streak") == 0).all()


def test_rest_is_capped(wide, default_cfg):
    played = wide.filter(pl.col("finished"))
    assert played.get_column("home_rest").max() <= default_cfg.rest_cap
    assert played.get_column("away_rest").max() <= default_cfg.rest_cap


# --- ELO / H2H ------------------------------------------------------------------

def test_elo_h2h_columns_present(wide):
    for col in ("home_elo", "away_elo", "elo_diff",
                "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd"):
        assert col in wide.columns


def test_elo_h2h_non_null_for_all_matches(wide):
    # contrairement à team_form, ELO/H2H reportent le record courant -> jamais null
    for col in ("home_elo", "away_elo", "elo_diff", "h2h_n"):
        assert wide.get_column(col).null_count() == 0


def test_elo_diff_is_consistent(wide, default_cfg):
    err = wide.select(
        (
            pl.col("elo_diff")
            - (pl.col("home_elo")
               + default_cfg.home_adv * (1 - pl.col("neutral").cast(pl.Int8))
               - pl.col("away_elo"))
        ).abs().max()
    ).item()
    assert err < 1e-9


def test_first_chronological_match_uses_base_elo(wide, default_cfg):
    first = wide.sort("date", "match_id").row(0, named=True)
    assert first["home_elo"] == default_cfg.elo_base
    assert first["away_elo"] == default_cfg.elo_base


def test_h2h_counts_non_negative(wide):
    assert wide.get_column("h2h_n").min() >= 0


def test_h2h_cold_start_uses_defaults(wide, default_cfg):
    cold = wide.filter(pl.col("h2h_n") == 0)
    assert cold.height > 0
    assert (cold.get_column("h2h_home_winrate") == default_cfg.h2h_default_winrate).all()
    assert (cold.get_column("h2h_draw_rate") == default_cfg.h2h_default_draw_rate).all()
    assert (cold.get_column("h2h_gd") == default_cfg.h2h_default_gd).all()
