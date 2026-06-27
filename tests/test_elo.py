"""Tests unitaires de l'ELO — fixtures synthétiques, valeurs exactes.

Les matchs sont joués sur terrain neutre (`neutral=True`) sauf mention contraire : l'avantage
du terrain est alors nul, donc deux équipes à 1500 ont une espérance de 0.5, ce qui rend les
deltas calculables à la main.
"""
from datetime import date

import polars as pl

from training.config import FeatureConfig
from training.features.elo import add_elo, _goal_diff_multiplier


def _elo(wide: pl.DataFrame, match_id: int, side: str) -> float:
    return wide.filter(pl.col("match_id") == match_id).get_column(f"{side}_elo").item()


# --- multiplicateur de buts ------------------------------------------------------

def test_goal_diff_multiplier_values(default_cfg):
    assert _goal_diff_multiplier(0, default_cfg) == 1.0
    assert _goal_diff_multiplier(1, default_cfg) == 1.0
    assert _goal_diff_multiplier(-2, default_cfg) == 1.5
    assert _goal_diff_multiplier(3, default_cfg) == (11 + 3) / 8  # 1.75


def test_goal_diff_multiplier_reads_config():
    # les constantes FIFA sont bien tunables via cfg
    cfg = FeatureConfig(gd_mult_medium=2.0, gd_mult_intercept=10.0, gd_mult_divisor=10.0)
    assert _goal_diff_multiplier(2, cfg) == 2.0
    assert _goal_diff_multiplier(3, cfg) == (10.0 + 3) / 10.0  # 1.3


# --- cold start / no-leakage -----------------------------------------------------

def test_first_match_uses_base_rating(make_res, default_cfg):
    res = make_res([(date(2020, 1, 1), "A", "B", 1, 0)])
    elo = add_elo(res, default_cfg)
    assert _elo(elo, 0, "home") == default_cfg.elo_base
    assert _elo(elo, 0, "away") == default_cfg.elo_base  # le résultat n'entre pas dans son propre rating


# --- delta exact + somme nulle ---------------------------------------------------

def test_exact_delta_and_zero_sum(make_res):
    cfg = FeatureConfig()  # k_base=30, cat 4 -> mult 1.0 -> K=30
    res = make_res([
        (date(2020, 1, 1), "A", "B", 1, 0),   # neutre, exp=0.5, s=1, g=1 -> delta = 15
        (date(2020, 1, 2), "A", "C", None, None),  # futur : A porte son rating courant
        (date(2020, 1, 3), "B", "D", None, None),  # futur : B porte son rating courant
    ], neutral=True)
    elo = add_elo(res, cfg)

    assert _elo(elo, 1, "home") == cfg.elo_base + 15.0  # A gagnant
    assert _elo(elo, 2, "home") == cfg.elo_base - 15.0  # B perdant
    # somme nulle : ce que A gagne, B le perd
    assert (_elo(elo, 1, "home") - cfg.elo_base) == -(_elo(elo, 2, "home") - cfg.elo_base)


# --- multiplicateur de buts câblé ------------------------------------------------

def test_larger_margin_moves_rating_more(make_res):
    cfg = FeatureConfig()
    narrow = make_res([
        (date(2020, 1, 1), "A", "B", 1, 0),       # g=1   -> delta=15
        (date(2020, 1, 2), "A", "C", None, None),
    ], neutral=True)
    wide = make_res([
        (date(2020, 1, 1), "A", "B", 3, 0),       # g=1.75 -> delta=26.25
        (date(2020, 1, 2), "A", "C", None, None),
    ], neutral=True)

    assert _elo(add_elo(narrow, cfg), 1, "home") == cfg.elo_base + 15.0
    assert _elo(add_elo(wide, cfg), 1, "home") == cfg.elo_base + 26.25


# --- K par catégorie -------------------------------------------------------------

def test_k_factor_scales_with_category(make_res):
    cfg = FeatureConfig()
    matches = [
        (date(2020, 1, 1), "A", "B", 1, 0),
        (date(2020, 1, 2), "A", "C", None, None),
    ]
    cat4 = make_res(matches, neutral=True, tournament_category=4)  # mult 1.0
    cat1 = make_res(matches, neutral=True, tournament_category=1)  # mult 2.0

    delta4 = _elo(add_elo(cat4, cfg), 1, "home") - cfg.elo_base
    delta1 = _elo(add_elo(cat1, cfg), 1, "home") - cfg.elo_base
    assert delta1 == 2.0 * delta4


# --- avantage du terrain ---------------------------------------------------------

def test_home_advantage_changes_expectation(make_res):
    cfg = FeatureConfig()
    # à domicile, A est favorisé (exp>0.5) -> gagner rapporte moins que sur terrain neutre
    home = make_res([
        (date(2020, 1, 1), "A", "B", 1, 0),
        (date(2020, 1, 2), "A", "C", None, None),
    ], neutral=False)
    neutral = make_res([
        (date(2020, 1, 1), "A", "B", 1, 0),
        (date(2020, 1, 2), "A", "C", None, None),
    ], neutral=True)

    delta_home = _elo(add_elo(home, cfg), 1, "home") - cfg.elo_base
    delta_neutral = _elo(add_elo(neutral, cfg), 1, "home") - cfg.elo_base
    assert delta_neutral == 15.0
    assert delta_home < delta_neutral


# --- échelle logistique ----------------------------------------------------------

def test_elo_scale_is_wired(make_res):
    # avec un écart de rating préexistant, l'espérance (donc le delta) dépend de elo_scale
    matches = [
        (date(2020, 1, 1), "A", "B", 5, 0),       # crée un écart A > B
        (date(2020, 1, 2), "A", "B", 1, 0),       # ici l'espérance dépend de l'échelle
        (date(2020, 1, 3), "A", "C", None, None),  # lecture du rating courant de A
    ]
    narrow = add_elo(make_res(matches, neutral=True), FeatureConfig(elo_scale=200.0))
    wide = add_elo(make_res(matches, neutral=True), FeatureConfig(elo_scale=400.0))
    assert _elo(narrow, 2, "home") != _elo(wide, 2, "home")


# --- report sur fixture futur ----------------------------------------------------

def test_future_match_carries_current_rating(make_res):
    cfg = FeatureConfig()
    res = make_res([
        (date(2020, 1, 1), "A", "B", 2, 0),
        (date(2020, 1, 2), "A", "B", 2, 0),
        (date(2030, 1, 1), "A", "B", None, None),  # fixture futur
    ], neutral=True)
    elo = add_elo(res, cfg)
    # le fixture futur hérite du rating courant : non-null, et = rating de A APRÈS le 2e match
    # joué (donc strictement supérieur au rating à l'entrée du 2e match : A a encore gagné).
    assert _elo(elo, 2, "home") is not None
    assert _elo(elo, 2, "home") > _elo(elo, 1, "home") > cfg.elo_base
    assert _elo(elo, 2, "away") < _elo(elo, 1, "away") < cfg.elo_base
