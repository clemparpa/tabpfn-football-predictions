"""Tests unitaires de la logique de features de forme — fixtures synthétiques.

Toutes les équipes "France" jouent ici à domicile contre des adversaires distincts
(qui n'ont qu'un seul match), donc les colonnes `home_*` du frame renvoyé sont les
features de France et les valeurs attendues sont calculables à la main.
"""
from datetime import date

import polars as pl

from training.config import FeatureConfig
from training.features.team_form import _to_long, add_team_form


def _home_col(wide: pl.DataFrame, name: str) -> list:
    return wide.sort("match_id").get_column(name).to_list()


# --- _to_long -------------------------------------------------------------------

def test_to_long_doubles_played_and_excludes_unplayed(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 2, 0),
        (date(2020, 1, 2), "C", "D", 1, 1),
        (date(2020, 1, 3), "E", "F", None, None),  # non joué -> exclu
    ])
    long = _to_long(res, default_cfg)

    assert long.height == 4  # 2 matchs joués * 2 perspectives
    assert long.get_column("is_home").sum() == 2  # moitié home, moitié away
    assert long.filter(pl.col("match_id") == 2).height == 0  # le non-joué a disparu


def test_to_long_points_won_draw_goal_diff(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "X", "Y", 2, 0),  # X gagne
        (date(2020, 1, 2), "X", "Z", 1, 1),  # X fait nul
        (date(2020, 1, 3), "X", "W", 0, 2),  # X perd
    ])
    long = _to_long(res, default_cfg).filter(pl.col("team") == "X").sort("date")

    assert long.get_column("points").to_list() == [3, 1, 0]
    assert long.get_column("won").to_list() == [1, 0, 0]
    assert long.get_column("draw").to_list() == [0, 1, 0]
    assert long.get_column("goal_diff").to_list() == [2, 0, -2]


# --- streaks --------------------------------------------------------------------

def test_win_streak_resets_after_non_win(make_res, default_cfg):
    # France : V, V, N, V  -> win_streak à l'entrée = [0, 1, 2, 0]
    res = make_res([
        (date(2020, 1, 1), "France", "A", 2, 0),
        (date(2020, 1, 2), "France", "B", 1, 0),
        (date(2020, 1, 3), "France", "C", 1, 1),
        (date(2020, 1, 4), "France", "D", 3, 0),
    ])
    wide = add_team_form(res, default_cfg)
    assert _home_col(wide, "home_win_streak") == [0, 1, 2, 0]


def test_draw_streak_resets_after_non_draw(make_res, default_cfg):
    # France : N, N, V, N  -> draw_streak à l'entrée = [0, 1, 2, 0]
    res = make_res([
        (date(2020, 1, 1), "France", "A", 1, 1),
        (date(2020, 1, 2), "France", "B", 0, 0),
        (date(2020, 1, 3), "France", "C", 2, 1),
        (date(2020, 1, 4), "France", "D", 1, 1),
    ])
    wide = add_team_form(res, default_cfg)
    assert _home_col(wide, "home_draw_streak") == [0, 1, 2, 0]


# --- no-leakage / rolling -------------------------------------------------------

def test_rolling_excludes_current_match(make_res, default_cfg):
    # France gagne 3 fois (3 pts each) -> points_history entrant = [default, 3, 3]
    res = make_res([
        (date(2020, 1, 1), "France", "A", 2, 0),
        (date(2020, 1, 2), "France", "B", 2, 0),
        (date(2020, 1, 3), "France", "C", 2, 0),
    ])
    wide = add_team_form(res, default_cfg)
    assert _home_col(wide, "home_points_history") == [default_cfg.default_points, 3.0, 3.0]


def test_window_size_is_wired_from_config(make_res):
    # France : V, D, V  (points 3, 0, 3). Au 3e match :
    #   - fenêtre 5 -> mean([3, 0]) = 1.5
    #   - fenêtre 1 -> mean([0])    = 0.0
    matches = [
        (date(2020, 1, 1), "France", "A", 2, 0),
        (date(2020, 1, 2), "France", "B", 0, 1),
        (date(2020, 1, 3), "France", "C", 2, 0),
    ]
    wide5 = add_team_form(make_res(matches), FeatureConfig(points_history_size=5))
    wide1 = add_team_form(make_res(matches), FeatureConfig(points_history_size=1))

    assert _home_col(wide5, "home_points_history")[2] == 1.5
    assert _home_col(wide1, "home_points_history")[2] == 0.0


# --- defaults cold-start --------------------------------------------------------

def test_cold_start_uses_config_defaults(make_res, default_cfg):
    res = make_res([(date(2020, 1, 1), "A", "B", 1, 0)])
    wide = add_team_form(res, default_cfg)
    row = wide.row(0, named=True)

    for side in ("home", "away"):
        assert row[f"{side}_points_history"] == default_cfg.default_points
        assert row[f"{side}_winrate_history"] == default_cfg.default_rate
        assert row[f"{side}_drawrate_history"] == default_cfg.default_rate
        assert row[f"{side}_team_score_history"] == default_cfg.default_scores
        assert row[f"{side}_opponent_score_history"] == default_cfg.default_scores
        assert row[f"{side}_goal_diff_history"] == default_cfg.default_goal_diff
        assert row[f"{side}_played"] == 0
        assert row[f"{side}_rest"] == default_cfg.default_rest
        assert row[f"{side}_win_streak"] == 0
        assert row[f"{side}_draw_streak"] == 0


# --- played & rest --------------------------------------------------------------

def test_played_counts_prior_matches(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "France", "A", 1, 0),
        (date(2020, 1, 2), "France", "B", 1, 0),
        (date(2020, 1, 3), "France", "C", 1, 0),
    ])
    wide = add_team_form(res, default_cfg)
    assert _home_col(wide, "home_played") == [0, 1, 2]


def test_rest_days_default_and_cap(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "France", "A", 1, 0),   # 1er match -> default_rest
        (date(2020, 1, 6), "France", "B", 1, 0),    # +5 jours
        (date(2020, 5, 1), "France", "C", 1, 0),    # +116 jours -> plafonné
    ])
    wide = add_team_form(res, default_cfg)
    rest = _home_col(wide, "home_rest")
    assert rest[0] == default_cfg.default_rest
    assert rest[1] == 5
    assert rest[2] == default_cfg.rest_cap
