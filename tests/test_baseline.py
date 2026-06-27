"""Tests du baseline (offline, sans réseau).

Garantit l'essentiel pour la comparaison : le baseline pandas sélectionne **les mêmes lignes de
test** que le pipeline polars (`make_backtest_split`), et borne correctement son pool
d'entraînement. Aucun appel TabPFN (seule la sélection/feature build est exercée).
"""
from collections import Counter
from datetime import date

import pandas as pd

import baseline
from training.backtest import make_backtest_split

CUTOFF = date(2018, 1, 1)
TRAIN_YEARS = 4


def _triples_pandas(df) -> Counter:
    return Counter(
        (pd.Timestamp(d).date(), h, a)
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"])
    )


def _triples_polars(df) -> Counter:
    return Counter(
        (d, h, a)
        for d, h, a in zip(
            df.get_column("date").to_list(),
            df.get_column("home_team").to_list(),
            df.get_column("away_team").to_list(),
        )
    )


def test_baseline_test_rows_match_pipeline(real_matches):
    feats = baseline.build_features(baseline.load_data())
    _, baseline_test = baseline.backtest_split(feats, pd.Timestamp(CUTOFF), TRAIN_YEARS)

    pipeline_test = make_backtest_split(CUTOFF, TRAIN_YEARS, res=real_matches).test

    assert _triples_pandas(baseline_test) == _triples_polars(pipeline_test)


def test_baseline_train_window_bounds():
    feats = baseline.build_features(baseline.load_data())
    cutoff = pd.Timestamp(CUTOFF)
    pool, _ = baseline.backtest_split(feats, cutoff, TRAIN_YEARS, max_train=baseline.MAX_TRAIN)

    train_start = cutoff - pd.DateOffset(years=TRAIN_YEARS)
    assert (pool["date"] >= train_start).all()
    assert (pool["date"] < cutoff).all()
    assert pool["outcome"].notna().all()
    assert len(pool) <= baseline.MAX_TRAIN


def test_baseline_max_train_caps_pool():
    feats = baseline.build_features(baseline.load_data())
    cap = 50
    pool, _ = baseline.backtest_split(feats, pd.Timestamp(CUTOFF), TRAIN_YEARS, max_train=cap)
    assert len(pool) == cap  # la fenêtre contient bien plus que `cap` matchs joués
