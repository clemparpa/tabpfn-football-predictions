"""Recherche d'hyperparamètres Optuna pour le backtest TabPFN.

On optimise le **log-loss multi-classe** (métrique du concours, *lower is better*) en faisant
varier conjointement quatre leviers par trial :

1. les hyperparamètres de `FeatureConfig` (fenêtres de forme, ELO, cold-starts H2H) ;
2. la profondeur d'historique d'entraînement `train_years` ;
3. la sélection de colonnes données au modèle, par **groupe** (`FEATURE_GROUPS`) ;
4. les hyperparamètres de `tabpfn_client.TabPFNClassifier` (standards + mode *thinking*).

Par défaut, on tourne en mode **lean** : TabPFN est figé (`LEAN_TABPFN_KWARGS`) et l'ELO est
réduit à ses 3 params les plus influents, pour concentrer le budget d'appels sur les leviers à
fort impact (colonnes, `train_years`, ELO de base) et converger vite. `--full` rouvre la
recherche large (tout le bloc TabPFN + ELO/cold-starts) — à réserver au micro-tuning du gagnant.

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
    date(2025, 1, 1),
)

# Groupe(s) de secours si le tirage désactive toutes les familles (TabPFN exige >=1 feature).
_FALLBACK_GROUPS: tuple[str, ...] = ("elo", "diffs")

# Config TabPFN figée en mode lean (défaut). TabPFN est un modèle pré-entraîné, robuste à ses
# hyperparamètres : les tuner coûte cher en latence/quota pour un gain marginal. On fige donc le
# bloc et on concentre la recherche sur les leviers spécifiques au problème (colonnes, train_years,
# ELO). `n_estimators=2` est le compromis latence/bruit ; le mettre à 1 accélère encore, l'augmenter
# lisse un peu les probas. `--full` rouvre la recherche complète sur TabPFN/ELO.
LEAN_TABPFN_KWARGS: dict = {"n_estimators": 2}


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


def suggest_feature_config(trial, *, team_form_active: bool, lean: bool = True) -> FeatureConfig:
    """Suggère un sous-ensemble curaté de champs `FeatureConfig` (le reste garde ses défauts).

    Les tailles de fenêtres `team_form` ne sont tirées **que si** le groupe est actif (une
    fenêtre partagée pour toutes les histoires, afin de réduire la dimensionnalité).

    En mode `lean` (défaut) on ne tire que les 3 params ELO les plus influents (`home_adv`,
    `k_base`, `elo_scale`) ; `gd_mult_*` et les cold-starts H2H restent à leurs défauts. Le mode
    complet (`lean=False`) rouvre tout le bloc.
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
    # ELO (les 3 leviers les plus influents, toujours tirés)
    params.update(
        home_adv=trial.suggest_float("home_adv", 0.0, 150.0),
        k_base=trial.suggest_float("k_base", 10.0, 60.0),
        elo_scale=trial.suggest_float("elo_scale", 200.0, 600.0),
    )
    if not lean:
        # ELO fin + cold-starts : seulement en recherche complète.
        params.update(
            gd_mult_medium=trial.suggest_float("gd_mult_medium", 1.0, 2.5),
            gd_mult_divisor=trial.suggest_float("gd_mult_divisor", 4.0, 12.0),
            default_points=trial.suggest_float("default_points", 0.5, 2.0),
            h2h_default_winrate=trial.suggest_float("h2h_default_winrate", 0.3, 0.7),
            h2h_default_draw_rate=trial.suggest_float("h2h_default_draw_rate", 0.1, 0.4),
        )
    return FeatureConfig(**params)


def suggest_tabpfn_kwargs(trial, *, include_thinking: bool, lean: bool = True) -> dict:
    """Suggère les kwargs passés à `TabPFNClassifier` (hors `random_state`/`ignore_*`).

    En mode `lean` (défaut) TabPFN est **figé** (`LEAN_TABPFN_KWARGS`) : rien n'est tiré. Le mode
    complet (`lean=False`) tire les params standards, et le bloc *thinking* si `include_thinking`
    ET que le trial l'active.
    """
    if lean:
        return dict(LEAN_TABPFN_KWARGS)
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
    lean: bool = True,
    classifier_factory=None,
    log_mlflow: bool = False,
    experiment: str = OPTUNA_EXPERIMENT,
) -> float:
    """Évalue un trial en walk-forward et renvoie le log-loss moyen (à minimiser).

    `lean` (défaut) restreint l'espace de recherche : ELO réduit à 3 params, TabPFN figé
    (cf. `suggest_feature_config`/`suggest_tabpfn_kwargs`). `lean=False` rouvre la recherche large.

    `classifier_factory(kwargs, random_state)` est injectable (défaut : `build_tabpfn`) — un faux
    classifieur permet de tester l'orchestration sans réseau. Une **instance fraîche par cutoff**
    garantit un refit propre.
    """
    factory = classifier_factory or build_tabpfn
    train_years = trial.suggest_int("train_years", *train_years_range)
    columns, active_groups = suggest_feature_columns(trial)
    cfg = suggest_feature_config(
        trial, team_form_active="team_form" in active_groups, lean=lean
    )
    tabpfn_kwargs = suggest_tabpfn_kwargs(trial, include_thinking=include_thinking, lean=lean)

    losses: list[float] = []
    accuracies: list[float] = []
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
        accuracies.append(result.accuracy)
        # On reporte la moyenne courante : le MedianPruner compare des trials au même step.
        trial.report(sum(losses) / len(losses), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_loss = sum(losses) / len(losses)
    mean_accuracy = sum(accuracies) / len(accuracies)
    # L'accuracy n'est pas l'objectif (on minimise le log-loss) mais on la garde comme métrique
    # de suivi, visible dans le dataframe de l'étude / le dashboard Optuna.
    trial.set_user_attr("accuracy", mean_accuracy)
    if log_mlflow:
        _log_trial_to_mlflow(
            trial, cfg, columns, tabpfn_kwargs, train_years, cutoffs,
            losses, mean_loss, accuracies, mean_accuracy, experiment,
        )
    return mean_loss


def _log_trial_to_mlflow(
    trial, cfg: FeatureConfig, columns, tabpfn_kwargs, train_years, cutoffs,
    losses, mean_loss, accuracies, mean_accuracy, experiment,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
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
        mlflow.log_metric("accuracy", mean_accuracy)
        for i, (loss, acc) in enumerate(zip(losses, accuracies)):
            mlflow.log_metric("log_loss_per_cutoff", loss, step=i)
            mlflow.log_metric("accuracy_per_cutoff", acc, step=i)


def run_study(
    *,
    n_trials: int,
    cutoffs,
    train_years_range: tuple[int, int] = TRAIN_YEARS_RANGE,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    categories=None,
    storage: str | None = OPTUNA_STORAGE,
    study_name: str = DEFAULT_STUDY_NAME,
    experiment: str | None = None,
    include_thinking: bool = True,
    lean: bool = True,
    random_state: int = 42,
    log_mlflow: bool = True,
    classifier_factory=None,
):
    """Crée/charge l'étude (résumable) et lance `n_trials` essais d'optimisation.

    `lean` (défaut) restreint l'espace de recherche (cf. `objective`). `classifier_factory` est
    passé à `objective` (défaut : `build_tabpfn`, appel réseau) — un faux classifieur permet un
    run hors-ligne.

    `experiment` nomme l'expérience MLflow ; par défaut elle est **dérivée du `study_name`**
    (`f"{study_name}-optuna"`) pour qu'une nouvelle étude ne soit pas écrasée dans le même bucket.
    """
    experiment = experiment or f"{study_name}-optuna"
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
            lean=lean,
            log_mlflow=log_mlflow,
            classifier_factory=classifier_factory,
            experiment=experiment,
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
    parser.add_argument(
        "--experiment", default=None,
        help="nom de l'expérience MLflow (défaut : <study-name>-optuna)",
    )
    parser.add_argument("--storage", default=OPTUNA_STORAGE)
    parser.add_argument(
        "--full", action="store_true",
        help="recherche large : tous params ELO/cold-starts + TabPFN complet (défaut : lean)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="exclut le mode thinking (pertinent uniquement avec --full)",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le logging MLflow")
    args = parser.parse_args()

    cutoffs = args.cutoffs or default_cutoffs(args.n_cutoffs)
    run_study(
        n_trials=args.n_trials,
        cutoffs=cutoffs,
        max_train=args.max_train,
        study_name=args.study_name,
        experiment=args.experiment,
        storage=args.storage,
        include_thinking=not args.no_thinking,
        lean=not args.full,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
