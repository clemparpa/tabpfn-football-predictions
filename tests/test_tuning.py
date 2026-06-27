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

from training.model import FEATURE_COLUMNS, FEATURE_GROUPS
from training.tuning import (
    _FALLBACK_GROUPS,
    default_cutoffs,
    objective,
    suggest_feature_columns,
    suggest_feature_config,
    suggest_tabpfn_kwargs,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ALL_GROUPS_ON = {f"use_{g}": True for g in FEATURE_GROUPS}
# Params non-fenêtre toujours tirés par suggest_feature_config.
_CFG_BASE = {
    "home_adv": 60.0,
    "k_base": 30.0,
    "elo_scale": 400.0,
    "gd_mult_medium": 1.5,
    "gd_mult_divisor": 8.0,
    "default_points": 1.3,
    "h2h_default_winrate": 0.5,
    "h2h_default_draw_rate": 0.25,
}


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
        optuna.trial.FixedTrial({"history_size": 9, **_CFG_BASE}), team_form_active=True
    )
    assert cfg.points_history_size == 9
    assert cfg.goal_diff_history_size == 9  # fenêtre partagée
    assert cfg.home_adv == 60.0


def test_window_sizes_keep_defaults_when_team_form_inactive():
    # "history_size" n'est pas fourni : il ne doit pas être demandé si team_form est inactif.
    cfg = suggest_feature_config(optuna.trial.FixedTrial(_CFG_BASE), team_form_active=False)
    assert cfg.points_history_size == 5  # défaut FeatureConfig inchangé


# --- kwargs TabPFN --------------------------------------------------------------

def test_tabpfn_kwargs_without_thinking():
    base = {
        "n_estimators": 4,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": True,
    }
    kwargs = suggest_tabpfn_kwargs(optuna.trial.FixedTrial(base), include_thinking=False)
    assert kwargs == base
    assert "thinking_mode" not in kwargs


def test_tabpfn_kwargs_with_thinking():
    base = {
        "n_estimators": 4,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": True,
        "thinking_mode": True,
        "thinking_effort": "high",
    }
    kwargs = suggest_tabpfn_kwargs(optuna.trial.FixedTrial(base), include_thinking=True)
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


def test_default_cutoffs_bounds():
    assert default_cutoffs(1) == default_cutoffs(3)[:1]
    with pytest.raises(ValueError):
        default_cutoffs(0)
