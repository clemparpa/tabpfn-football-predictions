"""Tests de la recherche Optuna (offline : faux classifieur injecté, aucun appel réseau).

Les helpers `suggest_*` sont validés via `optuna.trial.FixedTrial` (valeurs figées, pas de
sampler). L'`objective` est exercé de bout en bout avec un `classifier_factory` renvoyant un
`DummyClassifier` sklearn, sur les vraies données du split — sans toucher TabPFN.
"""
from datetime import date

import numpy as np
import optuna
import pytest
from sklearn.dummy import DummyClassifier

from training.config import FeatureConfig
from training.model import FEATURE_COLUMNS
from training.tournament import Tournament
from training.tuning import (
    DEFAULT_CUTOFF_POOL,
    TABPFN_KWARGS,
    default_cutoffs,
    objective,
    run_study,
    suggest_feature_config,
    suggest_trial_params,
    tournament_objective,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Champs réglables de FeatureConfig : fenêtre partagée + ELO (home_adv/k_base + k_mult par
# catégorie) + multiplicateurs de marge + barème de points. `elo_scale`/`elo_base` et le
# cold-start H2H ne sont volontairement plus tirés (quasi no-op après standardisation TabPFN).
_CFG_PARAMS = {
    "history_size": 9,
    "home_adv": 60.0,
    "k_base": 30.0,
    "gd_mult_small": 1.0,
    "gd_mult_medium": 1.5,
    "gd_mult_intercept": 11.0,
    "gd_mult_divisor": 8.0,
    "victory_points": 3,
    "draw_points": 1,
    "lose_points": 0,
    **{f"k_mult_cat{i}": v for i, v in zip(range(1, 8), (2.0, 1.5, 1.17, 1.0, 0.67, 0.5, 0.33))},
}


# --- leviers figés : colonnes + TabPFN ------------------------------------------

def test_suggest_trial_params_freezes_columns_and_tabpfn():
    params = {"train_years": 6, **_CFG_PARAMS}
    train_years, columns, cfg, tabpfn_kwargs = suggest_trial_params(
        optuna.trial.FixedTrial(params), train_years_range=(2, 10)
    )
    assert train_years == 6
    assert columns == FEATURE_COLUMNS  # toutes les colonnes, toujours
    assert tabpfn_kwargs == TABPFN_KWARGS
    assert tabpfn_kwargs is not TABPFN_KWARGS  # copie défensive
    assert cfg.k_base == 30.0  # FeatureConfig bien tiré


def test_tabpfn_kwargs_frozen_no_thinking():
    # TabPFN gelé : 2 estimators, aucun mode thinking.
    assert TABPFN_KWARGS == {"n_estimators": 2}
    assert "thinking_mode" not in TABPFN_KWARGS


# --- FeatureConfig --------------------------------------------------------------

def test_suggest_feature_config_tunes_all_fields():
    cfg = suggest_feature_config(optuna.trial.FixedTrial(_CFG_PARAMS))
    assert cfg.points_history_size == 9
    assert cfg.goal_diff_history_size == 9  # fenêtre partagée
    assert cfg.home_adv == 60.0
    assert cfg.k_base == 30.0
    assert cfg.gd_mult_small == 1.0
    assert cfg.gd_mult_intercept == 11.0
    assert cfg.victory_points == 3
    assert cfg.draw_points == 1
    assert cfg.k_mult_by_category == (2.0, 1.5, 1.17, 1.0, 0.67, 0.5, 0.33)
    # elo_scale/elo_base/h2h_default_gd ne sont plus tirés → ils gardent leurs défauts.
    assert cfg.elo_base == FeatureConfig().elo_base
    assert cfg.h2h_default_gd == FeatureConfig().h2h_default_gd


# --- objectif de bout en bout (offline) -----------------------------------------

def test_objective_runs_walkforward_offline():
    cutoffs = (date(2018, 1, 1), date(2019, 1, 1))
    calls: list[int] = []

    def fake_factory(kwargs, random_state):
        calls.append(1)
        return DummyClassifier(strategy="prior")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            cutoffs=cutoffs,
            train_years_range=(4, 4),  # déterministe, split modeste
            classifier_factory=fake_factory,
            log_mlflow=False,
        ),
        n_trials=1,
    )

    assert len(calls) == len(cutoffs)  # une instance fraîche par cutoff
    assert np.isfinite(study.best_value)
    assert study.best_value > 0
    # L'accuracy est tracée en user_attr (métrique de suivi, pas l'objectif).
    accuracy = study.best_trial.user_attrs["accuracy"]
    assert 0.0 <= accuracy <= 1.0


def test_default_cutoffs_bounds():
    pool_size = len(DEFAULT_CUTOFF_POOL)
    assert default_cutoffs(1) == DEFAULT_CUTOFF_POOL[:1]  # le plus récent en tête
    assert default_cutoffs(pool_size) == DEFAULT_CUTOFF_POOL  # borne haute = pool entier
    with pytest.raises(ValueError):
        default_cutoffs(0)
    with pytest.raises(ValueError):
        default_cutoffs(pool_size + 1)  # au-delà du pool


def test_tournament_objective_optimizes_loto_offline():
    """`tournament_objective` renvoie le log-loss LOTO, un fit par tournoi, sans réseau."""
    pool = (Tournament("FIFA World Cup", 2014), Tournament("FIFA World Cup", 2018))
    calls: list[int] = []

    def fake_factory(kwargs, random_state):
        calls.append(1)
        return DummyClassifier(strategy="prior")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: tournament_objective(
            trial,
            tournaments=pool,
            train_years_range=(8, 8),  # déterministe
            classifier_factory=fake_factory,
            log_mlflow=False,
        ),
        n_trials=1,
    )

    assert len(calls) == len(pool)  # un fit par tournoi
    assert np.isfinite(study.best_value) and study.best_value > 0
    assert 0.0 <= study.best_trial.user_attrs["accuracy"] <= 1.0
    assert "FIFA World Cup 2018" in study.best_trial.user_attrs["per_tournament"]


def test_run_study_requires_exactly_one_metric():
    """`run_study` exige soit `cutoffs`, soit `tournaments` — pas les deux, pas aucun."""
    with pytest.raises(ValueError, match="exactement l'un"):
        run_study(n_trials=1, storage=None, log_mlflow=False)  # aucun
    with pytest.raises(ValueError, match="exactement l'un"):
        run_study(
            n_trials=1, cutoffs=(date(2018, 1, 1),),
            tournaments=(Tournament("FIFA World Cup", 2018),),
            storage=None, log_mlflow=False,
        )  # les deux
