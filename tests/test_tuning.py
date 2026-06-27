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
from training.model import FEATURE_COLUMNS, FEATURE_GROUPS
from training.tuning import (
    _FALLBACK_GROUPS,
    DEFAULT_CUTOFF_POOL,
    LEAN_TABPFN_KWARGS,
    default_cutoffs,
    objective,
    suggest_feature_columns,
    suggest_feature_config,
    suggest_tabpfn_kwargs,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ALL_GROUPS_ON = {f"use_{g}": True for g in FEATURE_GROUPS}
# Les 3 params ELO toujours tirés (en lean comme en full).
_ELO_BASE = {"home_adv": 60.0, "k_base": 30.0, "elo_scale": 400.0}
# Bloc supplémentaire tiré uniquement en mode full (gd_mult_* + cold-starts).
_FULL_EXTRA = {
    "gd_mult_medium": 1.5,
    "gd_mult_divisor": 8.0,
    "default_points": 1.3,
    "h2h_default_winrate": 0.5,
    "h2h_default_draw_rate": 0.25,
}
_CFG_BASE = {**_ELO_BASE, **_FULL_EXTRA}


# --- sélection de colonnes ------------------------------------------------------

def test_all_groups_on_yields_full_columns():
    columns, active = suggest_feature_columns(optuna.trial.FixedTrial(_ALL_GROUPS_ON))
    assert columns == FEATURE_COLUMNS  # même contenu/ordre que la définition complète
    assert set(active) == set(FEATURE_GROUPS)


def test_subset_of_groups_selects_their_columns():
    toggles = {f"use_{g}": (g in {"elo", "h2h"}) for g in FEATURE_GROUPS}
    columns, active = suggest_feature_columns(optuna.trial.FixedTrial(toggles))
    assert set(active) == {"elo", "h2h"}
    assert set(columns) == set(FEATURE_GROUPS["elo"] + FEATURE_GROUPS["h2h"])


def test_all_groups_off_falls_back():
    toggles = {f"use_{g}": False for g in FEATURE_GROUPS}
    columns, active = suggest_feature_columns(optuna.trial.FixedTrial(toggles))
    assert active == _FALLBACK_GROUPS  # garde anti-vide : >=1 feature pour TabPFN
    assert len(columns) > 0


# --- FeatureConfig --------------------------------------------------------------

def test_window_sizes_tuned_only_when_team_form_active():
    cfg = suggest_feature_config(
        optuna.trial.FixedTrial({"history_size": 9, **_ELO_BASE}), team_form_active=True
    )
    assert cfg.points_history_size == 9
    assert cfg.goal_diff_history_size == 9  # fenêtre partagée
    assert cfg.home_adv == 60.0


def test_window_sizes_keep_defaults_when_team_form_inactive():
    # "history_size" n'est pas fourni : il ne doit pas être demandé si team_form est inactif.
    cfg = suggest_feature_config(optuna.trial.FixedTrial(_ELO_BASE), team_form_active=False)
    assert cfg.points_history_size == 5  # défaut FeatureConfig inchangé


def test_lean_config_skips_gd_mult_and_cold_starts():
    # En lean, seuls les 3 params ELO sont tirés : un FixedTrial sans le bloc full ne lève pas,
    # et gd_mult_*/cold-starts gardent leurs défauts FeatureConfig.
    default = FeatureConfig()
    cfg = suggest_feature_config(optuna.trial.FixedTrial(_ELO_BASE), team_form_active=False)
    assert cfg.k_base == 30.0  # tiré
    assert cfg.gd_mult_medium == default.gd_mult_medium  # figé au défaut
    assert cfg.h2h_default_winrate == default.h2h_default_winrate


def test_full_config_tunes_gd_mult_and_cold_starts():
    cfg = suggest_feature_config(
        optuna.trial.FixedTrial(_CFG_BASE), team_form_active=False, lean=False
    )
    assert cfg.gd_mult_medium == 1.5
    assert cfg.default_points == 1.3
    assert cfg.h2h_default_winrate == 0.5


# --- kwargs TabPFN --------------------------------------------------------------

def test_lean_tabpfn_kwargs_are_frozen():
    # En lean (défaut) rien n'est tiré : un FixedTrial vide suffit et on récupère la config figée.
    kwargs = suggest_tabpfn_kwargs(optuna.trial.FixedTrial({}), include_thinking=True)
    assert kwargs == LEAN_TABPFN_KWARGS
    assert kwargs is not LEAN_TABPFN_KWARGS  # copie défensive


def test_full_tabpfn_kwargs_without_thinking():
    base = {
        "n_estimators": 4,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": True,
    }
    kwargs = suggest_tabpfn_kwargs(
        optuna.trial.FixedTrial(base), include_thinking=False, lean=False
    )
    assert kwargs == base
    assert "thinking_mode" not in kwargs


def test_full_tabpfn_kwargs_with_thinking():
    base = {
        "n_estimators": 4,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": True,
        "thinking_mode": True,
        "thinking_effort": "high",
    }
    kwargs = suggest_tabpfn_kwargs(
        optuna.trial.FixedTrial(base), include_thinking=True, lean=False
    )
    assert kwargs["thinking_mode"] is True
    assert kwargs["thinking_effort"] == "high"


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
            include_thinking=True,
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
