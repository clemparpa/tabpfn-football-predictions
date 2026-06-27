"""Recherche d'hyperparamètres Optuna pour le backtest TabPFN.

On optimise le **log-loss multi-classe** (métrique du concours, *lower is better*) en faisant
varier conjointement quatre leviers par trial :

1. les hyperparamètres de `FeatureConfig` (fenêtres de forme, ELO, cold-starts H2H) ;
2. la profondeur d'historique d'entraînement `train_years` ;
3. la sélection de colonnes données au modèle, par **groupe** (`FEATURE_GROUPS`) ;
4. les hyperparamètres de `tabpfn_client.TabPFNClassifier` (standards + mode *thinking*).

Chaque trial est évalué en **walk-forward** : on moyenne le log-loss sur une liste de `cutoffs`
configurables (1 cutoff = exploration rapide ; 3-5 = estimation plus robuste). Comme chaque
cutoff déclenche un entraînement TabPFN **distant** (lent, sur quota d'API), un `MedianPruner`
coupe les trials faibles avant d'épuiser tous les cutoffs.

L'objectif est testable **hors-ligne** : `classifier_factory` est injectable (un faux classifieur
sklearn évite tout appel réseau), à l'image de `classifier=` dans `training.model`.

Le suivi se fait via un store SQLite Optuna (`sqlite:///optuna.db`) — résumable et
parallélisable (`load_if_exists`) — et, optionnellement, un run MLflow par trial.
"""
import argparse
from dataclasses import asdict
from datetime import date

import optuna

from training.config import FeatureConfig
from training.model import (
    DEFAULT_MAX_TRAIN,
    FEATURE_GROUPS,
    MLFLOW_TRACKING_URI,
    run_backtest,
)

OPTUNA_STORAGE = "sqlite:///optuna.db"
DEFAULT_STUDY_NAME = "tabpfn-football"
OPTUNA_EXPERIMENT = "tabpfn-football-optuna"  # expérience MLflow dédiée à la recherche
TRAIN_YEARS_RANGE = (4, 10)

# Dates de cutoff candidates (plus récente en tête) ; `default_cutoffs(n)` en prend les n
# premières. Chaque cutoff masque tout match >= date et teste sur les matchs réellement joués.
DEFAULT_CUTOFF_POOL: tuple[date, ...] = (
    date(2026, 1, 2),
    date(2025, 1, 1),
)

# Groupe(s) de secours si le tirage désactive toutes les familles (TabPFN exige >=1 feature).
_FALLBACK_GROUPS: tuple[str, ...] = ("elo", "diffs")


def default_cutoffs(n: int) -> tuple[date, ...]:
    """Renvoie les `n` cutoffs les plus récents du pool."""
    if not 1 <= n <= len(DEFAULT_CUTOFF_POOL):
        raise ValueError(f"n_cutoffs doit être dans [1, {len(DEFAULT_CUTOFF_POOL)}], reçu {n}")
    return DEFAULT_CUTOFF_POOL[:n]


# --- Espace de recherche : un helper pur par levier (chacun prend un `trial`) -----------

def suggest_feature_columns(trial) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Toggle on/off chaque groupe de `FEATURE_GROUPS`.

    Renvoie `(colonnes, groupes_actifs)`. Garantit au moins un groupe actif : si le tirage
    désactive tout, on retombe sur `_FALLBACK_GROUPS` (TabPFN exige >=1 colonne).
    """
    active = [g for g in FEATURE_GROUPS if trial.suggest_categorical(f"use_{g}", [True, False])]
    if not active:
        active = list(_FALLBACK_GROUPS)
    columns = tuple(c for g in FEATURE_GROUPS if g in active for c in FEATURE_GROUPS[g])
    return columns, tuple(active)


def suggest_feature_config(trial, *, team_form_active: bool) -> FeatureConfig:
    """Suggère un sous-ensemble curaté de champs `FeatureConfig` (le reste garde ses défauts).

    Les tailles de fenêtres `team_form` ne sont tirées **que si** le groupe est actif (une
    fenêtre partagée pour toutes les histoires, afin de réduire la dimensionnalité).
    """
    params: dict = {}
    if team_form_active:
        window = trial.suggest_int("history_size", 3, 15)
        params.update(
            points_history_size=window,
            winrate_history_size=window,
            drawrate_history_size=window,
            scores_history_size=window,
            goal_diff_history_size=window,
        )
    # ELO
    params.update(
        home_adv=trial.suggest_float("home_adv", 0.0, 150.0),
        k_base=trial.suggest_float("k_base", 10.0, 60.0),
        elo_scale=trial.suggest_float("elo_scale", 200.0, 600.0),
        gd_mult_medium=trial.suggest_float("gd_mult_medium", 1.0, 2.5),
        gd_mult_divisor=trial.suggest_float("gd_mult_divisor", 4.0, 12.0),
    )
    # Cold-starts
    params.update(
        default_points=trial.suggest_float("default_points", 0.5, 2.0),
        h2h_default_winrate=trial.suggest_float("h2h_default_winrate", 0.3, 0.7),
        h2h_default_draw_rate=trial.suggest_float("h2h_default_draw_rate", 0.1, 0.4),
    )
    return FeatureConfig(**params)


def suggest_tabpfn_kwargs(trial, *, include_thinking: bool) -> dict:
    """Suggère les kwargs passés à `TabPFNClassifier` (hors `random_state`/`ignore_*`).

    Bloc *thinking* tiré seulement si `include_thinking` ET que le trial l'active.
    """
    kwargs = dict(
        n_estimators=trial.suggest_int("n_estimators", 1, 8),
        softmax_temperature=trial.suggest_float("softmax_temperature", 0.5, 1.5),
        balance_probabilities=trial.suggest_categorical("balance_probabilities", [True, False]),
        average_before_softmax=trial.suggest_categorical("average_before_softmax", [True, False]),
    )
    if include_thinking and trial.suggest_categorical("thinking_mode", [True, False]):
        kwargs["thinking_mode"] = True
        kwargs["thinking_effort"] = trial.suggest_categorical("thinking_effort", ["medium", "high"])
    return kwargs


def build_tabpfn(kwargs: dict, random_state: int):
    """Construit un `TabPFNClassifier` distant (import paresseux pour rester hors-ligne ailleurs)."""
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier(
        ignore_pretraining_limits=True, random_state=random_state, **kwargs
    )


# --- Objectif Optuna ---------------------------------------------------------------------

def objective(
    trial,
    *,
    cutoffs,
    train_years_range: tuple[int, int] = TRAIN_YEARS_RANGE,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    categories=None,
    random_state: int = 42,
    include_thinking: bool = True,
    classifier_factory=None,
    log_mlflow: bool = False,
) -> float:
    """Évalue un trial en walk-forward et renvoie le log-loss moyen (à minimiser).

    `classifier_factory(kwargs, random_state)` est injectable (défaut : `build_tabpfn`) — un faux
    classifieur permet de tester l'orchestration sans réseau. Une **instance fraîche par cutoff**
    garantit un refit propre.
    """
    factory = classifier_factory or build_tabpfn
    train_years = trial.suggest_int("train_years", *train_years_range)
    columns, active_groups = suggest_feature_columns(trial)
    cfg = suggest_feature_config(trial, team_form_active="team_form" in active_groups)
    tabpfn_kwargs = suggest_tabpfn_kwargs(trial, include_thinking=include_thinking)

    losses: list[float] = []
    for step, cutoff in enumerate(cutoffs):
        classifier = factory(tabpfn_kwargs, random_state)
        result = run_backtest(
            cutoff,
            train_years,
            categories,
            cfg,
            random_state=random_state,
            max_train=max_train,
            classifier=classifier,
            log_mlflow=False,
            feature_columns=columns,
        )
        losses.append(result.log_loss)
        # On reporte la moyenne courante : le MedianPruner compare des trials au même step.
        trial.report(sum(losses) / len(losses), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_loss = sum(losses) / len(losses)
    if log_mlflow:
        _log_trial_to_mlflow(trial, cfg, columns, tabpfn_kwargs, train_years, cutoffs, losses, mean_loss)
    return mean_loss


def _log_trial_to_mlflow(
    trial, cfg: FeatureConfig, columns, tabpfn_kwargs, train_years, cutoffs, losses, mean_loss
) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(OPTUNA_EXPERIMENT)
    with mlflow.start_run(run_name=f"trial-{trial.number}"):
        params = {f"cfg.{k}": str(v) for k, v in asdict(cfg).items()}
        params.update({f"tabpfn.{k}": str(v) for k, v in tabpfn_kwargs.items()})
        params.update(
            train_years=train_years,
            n_features=len(columns),
            n_cutoffs=len(cutoffs),
            feature_columns=str(list(columns)),
        )
        mlflow.log_params(params)
        mlflow.log_metric("log_loss", mean_loss)
        for i, loss in enumerate(losses):
            mlflow.log_metric("log_loss_per_cutoff", loss, step=i)


def run_study(
    *,
    n_trials: int,
    cutoffs,
    train_years_range: tuple[int, int] = TRAIN_YEARS_RANGE,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    categories=None,
    storage: str | None = OPTUNA_STORAGE,
    study_name: str = DEFAULT_STUDY_NAME,
    include_thinking: bool = True,
    random_state: int = 42,
    log_mlflow: bool = True,
    classifier_factory=None,
):
    """Crée/charge l'étude (résumable) et lance `n_trials` essais d'optimisation.

    `classifier_factory` est passé à `objective` (défaut : `build_tabpfn`, appel réseau) — un faux
    classifieur permet un run hors-ligne.
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            cutoffs=cutoffs,
            train_years_range=train_years_range,
            max_train=max_train,
            categories=categories,
            random_state=random_state,
            include_thinking=include_thinking,
            log_mlflow=log_mlflow,
            classifier_factory=classifier_factory,
        ),
        n_trials=n_trials,
    )
    print(f"best log_loss = {study.best_value:.4f}")
    print(f"best params   = {study.best_params}")
    return study


def _parse_cutoffs(value: str) -> tuple[date, ...]:
    return tuple(date.fromisoformat(x.strip()) for x in value.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-cutoffs", type=int, default=1, help="nb de cutoffs récents du pool")
    parser.add_argument(
        "--cutoffs", type=_parse_cutoffs, default=None,
        help="liste explicite, ex. 2024-01-01,2025-01-01 (prioritaire sur --n-cutoffs)",
    )
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", default=OPTUNA_STORAGE)
    parser.add_argument("--no-thinking", action="store_true", help="exclut le mode thinking")
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le logging MLflow")
    args = parser.parse_args()

    cutoffs = args.cutoffs or default_cutoffs(args.n_cutoffs)
    run_study(
        n_trials=args.n_trials,
        cutoffs=cutoffs,
        max_train=args.max_train,
        study_name=args.study_name,
        storage=args.storage,
        include_thinking=not args.no_thinking,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
