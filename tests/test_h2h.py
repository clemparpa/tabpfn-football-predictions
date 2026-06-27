"""Tests unitaires du Head-to-Head — fixtures synthétiques, valeurs exactes.

`pair_a`/`pair_b` étant la paire triée alphabétiquement, "A" est toujours `pair_a` face à
"B" ; les bilans canoniques sont donc calculables à la main quelle que soit l'équipe qui
reçoit.
"""
from datetime import date

import polars as pl

from training.features.h2h import add_h2h


def _h2h(wide: pl.DataFrame, match_id: int, col: str):
    return wide.filter(pl.col("match_id") == match_id).get_column(col).item()


# --- cold start ------------------------------------------------------------------

def test_first_meeting_uses_defaults(make_res, default_cfg):
    res = make_res([(date(2020, 1, 1), "A", "B", 1, 0)])
    h2h = add_h2h(res, default_cfg)
    assert _h2h(h2h, 0, "h2h_n") == 0  # le match courant n'est pas compté dans son propre bilan
    assert _h2h(h2h, 0, "h2h_home_winrate") == default_cfg.h2h_default_winrate
    assert _h2h(h2h, 0, "h2h_draw_rate") == default_cfg.h2h_default_draw_rate
    assert _h2h(h2h, 0, "h2h_gd") == default_cfg.h2h_default_gd


# --- accumulation et perspective -------------------------------------------------

def test_record_from_home_perspective(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 2, 0),  # A bat B
        (date(2020, 1, 2), "A", "B", 1, 0),  # re-match, A à domicile
    ])
    h2h = add_h2h(res, default_cfg)
    assert _h2h(h2h, 1, "h2h_n") == 1
    assert _h2h(h2h, 1, "h2h_home_winrate") == 1.0
    assert _h2h(h2h, 1, "h2h_draw_rate") == 0.0
    assert _h2h(h2h, 1, "h2h_gd") == 2.0


def test_perspective_flips_when_home_swaps(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 2, 0),  # A bat B
        (date(2020, 1, 2), "B", "A", 0, 0),  # re-match, B à domicile
    ])
    h2h = add_h2h(res, default_cfg)
    # du point de vue de B (à domicile) : il avait perdu, gd défavorable
    assert _h2h(h2h, 1, "h2h_n") == 1
    assert _h2h(h2h, 1, "h2h_home_winrate") == 0.0
    assert _h2h(h2h, 1, "h2h_gd") == -2.0


def test_draw_rate(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 1, 1),  # nul
        (date(2020, 1, 2), "A", "B", 1, 0),
    ])
    h2h = add_h2h(res, default_cfg)
    assert _h2h(h2h, 1, "h2h_draw_rate") == 1.0
    assert _h2h(h2h, 1, "h2h_home_winrate") == 0.0


def test_pair_key_is_symmetric(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 1, 0),  # A bat B (A à domicile)
        (date(2020, 1, 2), "B", "A", 1, 0),  # B bat A (B à domicile)
        (date(2020, 1, 3), "A", "B", 1, 0),  # 3e rencontre : doit compter les 2 précédentes
    ])
    h2h = add_h2h(res, default_cfg)
    assert _h2h(h2h, 2, "h2h_n") == 2  # même paire malgré l'inversion domicile/extérieur
    # du point de vue de A : 1 victoire (match0) sur 2 -> 0.5
    assert _h2h(h2h, 2, "h2h_home_winrate") == 0.5


# --- report sur fixture futur ----------------------------------------------------

def test_future_match_inherits_record(make_res, default_cfg):
    res = make_res([
        (date(2020, 1, 1), "A", "B", 3, 0),
        (date(2030, 1, 1), "A", "B", None, None),  # fixture futur
    ])
    h2h = add_h2h(res, default_cfg)
    assert _h2h(h2h, 1, "h2h_n") == 1
    assert _h2h(h2h, 1, "h2h_home_winrate") == 1.0
    assert _h2h(h2h, 1, "h2h_gd") == 3.0
