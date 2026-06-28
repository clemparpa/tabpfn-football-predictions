"""Recherche d'hyperparamètres Optuna pour le backtest TabPFN.

On optimise le **log-loss multi-classe** (métrique du concours, *lower is better*) en faisant
varier deux leviers par trial :

1. **tous** les hyperparamètres de `FeatureConfig` (fenêtres de forme, ELO complet, cold-starts) ;
2. la profondeur d'historique d'entraînement `train_years`.

Deux choix volontairement **figés** (pas tirés) : on donne toujours **toutes** les colonnes au
modèle (`FEATURE_COLUMNS`) et TabPFN est gelé sur `TABPFN_KWARGS` (`n_estimators=2`, pas de
*thinking*). TabPFN est un modèle pré-entraîné robuste à ses hyperparamètres : les tuner coûte
cher en latence/quota pour un gain marginal. Toute la recherche se concentre donc sur le
*feature engineering* (poids ELO, fenêtres) — le levier à fort impact sur ce problème.

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
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    run_backtest,
    train_classifier,
)
from training.tournament import DEFAULT_TOURNAMENTS, evaluate_tournaments, parse_tournaments

OPTUNA_STORAGE = "sqlite:///optuna.db"
DEFAULT_STUDY_NAME = "tabpfn-football"
OPTUNA_EXPERIMENT = "tabpfn-football-optuna"  # expérience MLflow dédiée à la recherche
TRAIN_YEARS_RANGE = (2, 10)

# Dates de cutoff candidates (plus récente en tête) ; `default_cutoffs(n)` en prend les n
# premières. Chaque cutoff masque tout match >= date et teste sur les matchs réellement joués.
DEFAULT_CUTOFF_POOL: tuple[date, ...] = (
    date(2025, 1, 1),
)

# Config TabPFN **figée** (jamais tirée). TabPFN est un modèle pré-entraîné, robuste à ses
# hyperparamètres : les tuner coûte cher en latence/quota pour un gain marginal. `n_estimators=2`
# est le compromis latence/bruit (1 accélère, plus lisse un peu les probas) ; pas de mode
# *thinking*. Toute la recherche porte sur `FeatureConfig` + `train_years`.
TABPFN_KWARGS: dict = {"n_estimators": 2}


def default_cutoffs(n: int) -> tuple[date, ...]:
    """Renvoie les `n` cutoffs les plus récents du pool."""
    if not 1 <= n <= len(DEFAULT_CUTOFF_POOL):
        raise ValueError(f"n_cutoffs doit être dans [1, {len(DEFAULT_CUTOFF_POOL)}], reçu {n}")
    return DEFAULT_CUTOFF_POOL[:n]


# --- Espace de recherche : un helper pur par levier (chacun prend un `trial`) -----------

def suggest_feature_config(trial) -> FeatureConfig:
    """Suggère **tous** les champs réglables de `FeatureConfig` (le reste garde ses défauts).

    Une fenêtre partagée pour toutes les histoires `team_form` (réduit la dimensionnalité), le bloc
    ELO complet (`home_adv`/`k_base`/`elo_scale`/`elo_base` + tous les `gd_mult_*` + les 7
    multiplicateurs `k_mult_by_category`), le barème de points (`victory`/`draw`/`lose`) et le
    cold-start H2H de différence de buts (`h2h_default_gd`).

    Note : `elo_base` (décalage commun à toutes les équipes) et l'échelle du barème de points sont
    quasi sans effet après standardisation des features par TabPFN — surtout du budget de recherche.
    """
    window = trial.suggest_int("history_size", 3, 15)
    k_mult = tuple(trial.suggest_float(f"k_mult_cat{i}", 0.1, 2.5) for i in range(1, 8))
    return FeatureConfig(
        points_history_size=window,
        winrate_history_size=window,
        drawrate_history_size=window,
        scores_history_size=window,
        goal_diff_history_size=window,
        # ELO : leviers de base + échelle de départ
        home_adv=trial.suggest_float("home_adv", 0.0, 150.0),
        k_base=trial.suggest_float("k_base", 10.0, 60.0),
        # elo_scale=trial.suggest_float("elo_scale", 200.0, 600.0),
        # elo_base=trial.suggest_float("elo_base", 1000.0, 2000.0),
        k_mult_by_category=k_mult,
        # multiplicateurs de marge (buts) : tous les seuils
        gd_mult_small=trial.suggest_float("gd_mult_small", 0.5, 1.5),
        gd_mult_medium=trial.suggest_float("gd_mult_medium", 1.0, 2.5),
        gd_mult_intercept=trial.suggest_float("gd_mult_intercept", 5.0, 15.0),
        gd_mult_divisor=trial.suggest_float("gd_mult_divisor", 4.0, 12.0),
        # barème de points (feature team_form) + cold-start H2H
        victory_points=trial.suggest_int("victory_points", 1, 5),
        draw_points=trial.suggest_int("draw_points", 0, 3),
        lose_points=trial.suggest_int("lose_points", 0, 2),
    )


def build_tabpfn(kwargs: dict, random_state: int):
    """Construit un `TabPFNClassifier` distant (import paresseux pour rester hors-ligne ailleurs)."""
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier(
        ignore_pretraining_limits=True, random_state=random_state, **kwargs
    )


def suggest_trial_params(trial, *, train_years_range: tuple[int, int]):
    """Tire les leviers d'un trial : `(train_years, colonnes, FeatureConfig, tabpfn_kwargs)`.

    Partagé par l'objectif calendaire (`objective`) et l'objectif tournoi (`tournament_objective`)
    pour qu'ils explorent **le même espace**. Les colonnes (toutes : `FEATURE_COLUMNS`) et les
    kwargs TabPFN (`TABPFN_KWARGS`) sont **figés** — seuls `FeatureConfig` et `train_years` varient.
    """
    train_years = trial.suggest_int("train_years", *train_years_range)
    cfg = suggest_feature_config(trial)
    return train_years, FEATURE_COLUMNS, cfg, dict(TABPFN_KWARGS)


# --- Objectif Optuna ---------------------------------------------------------------------

def objective(
    trial,
    *,
    cutoffs,
    train_years_range: tuple[int, int] = TRAIN_YEARS_RANGE,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    categories=None,
    random_state: int = 42,
    classifier_factory=None,
    log_mlflow: bool = False,
    experiment: str = OPTUNA_EXPERIMENT,
) -> float:
    """Évalue un trial en walk-forward et renvoie le log-loss moyen (à minimiser).

    On tire tout `FeatureConfig` + `train_years` ; colonnes et TabPFN sont figés
    (cf. `suggest_trial_params`).

    `classifier_factory(kwargs, random_state)` est injectable (défaut : `build_tabpfn`) — un faux
    classifieur permet de tester l'orchestration sans réseau. Une **instance fraîche par cutoff**
    garantit un refit propre.
    """
    factory = classifier_factory or build_tabpfn
    train_years, columns, cfg, tabpfn_kwargs = suggest_trial_params(
        trial, train_years_range=train_years_range
    )

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


def tournament_objective(
    trial,
    *,
    tournaments=DEFAULT_TOURNAMENTS,
    train_years_range: tuple[int, int] = TRAIN_YEARS_RANGE,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    random_state: int = 42,
    classifier_factory=None,
    log_mlflow: bool = False,
    experiment: str = OPTUNA_EXPERIMENT,
) -> float:
    """Évalue un trial sur le **backtest tournoi** et renvoie le log-loss LOTO (à minimiser).

    Même espace de recherche que `objective` (via `suggest_trial_params`) mais la métrique est la
    moyenne leave-one-tournament-out de `training.tournament.evaluate_tournaments` — proche de la
    cible (CDM 2026, neutre) plutôt que du log-loss calendaire global.

    `train_years` (tiré par le trial) borne l'historique d'entraînement de chaque tournoi.
    `classifier_factory(kwargs, random_state)` est injectable (défaut : `build_tabpfn`) — un faux
    classifieur teste l'orchestration sans réseau. Chaque tournoi = un fit (cf. `evaluate_tournaments`).
    """
    factory = classifier_factory or build_tabpfn
    train_years, columns, cfg, tabpfn_kwargs = suggest_trial_params(
        trial, train_years_range=train_years_range
    )

    def fit_fn(train, feature_columns):
        return train_classifier(train, factory(tabpfn_kwargs, random_state), random_state, feature_columns)

    report = evaluate_tournaments(
        tournaments,
        fit_fn=fit_fn,
        feature_columns=columns,
        cfg=cfg,
        max_train=max_train,
        train_years=train_years,
    )
    trial.set_user_attr("accuracy", report.loto_accuracy)
    trial.set_user_attr(
        "per_tournament",
        str([(r["tournament"], round(r["log_loss"], 4)) for r in report.per_tournament]),
    )
    if log_mlflow:
        _log_tournament_trial_to_mlflow(
            trial, cfg, columns, tabpfn_kwargs, train_years, report, experiment
        )
    return report.loto_log_loss


def _log_tournament_trial_to_mlflow(
    trial, cfg: FeatureConfig, columns, tabpfn_kwargs, train_years, report, experiment,
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
            n_tournaments=len(report.per_tournament),
            feature_columns=str(list(columns)),
        )
        mlflow.log_params(params)
        mlflow.log_metric("log_loss", report.loto_log_loss)  # = log-loss LOTO
        mlflow.log_metric("accuracy", report.loto_accuracy)
        for i, r in enumerate(report.per_tournament):
            mlflow.log_metric("log_loss_per_tournament", r["log_loss"], step=i)
            mlflow.log_metric("accuracy_per_tournament", r["accuracy"], step=i)


def run_study(
    *,
    n_trials: int,
    cutoffs=None,
    tournaments=None,
    train_years_range: tuple[int, int] = TRAIN_YEARS_RANGE,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    categories=None,
    storage: str | None = OPTUNA_STORAGE,
    study_name: str = DEFAULT_STUDY_NAME,
    experiment: str | None = None,
    random_state: int = 42,
    log_mlflow: bool = True,
    classifier_factory=None,
    n_jobs: int = 1,
):
    """Crée/charge l'étude (résumable) et lance `n_trials` essais d'optimisation.

    Deux métriques mutuellement exclusives : passer `cutoffs` optimise le **log-loss calendaire**
    (`objective`) ; passer `tournaments` optimise le **log-loss LOTO tournoi** (`tournament_objective`,
    proche de la cible CDM). Exactement l'une des deux doit être fournie.

    `classifier_factory` est passé à l'objectif (défaut : `build_tabpfn`, appel réseau) — un faux
    classifieur permet un run hors-ligne.

    `experiment` nomme l'expérience MLflow ; par défaut elle est **dérivée du `study_name`**
    (`f"{study_name}-optuna"`) pour qu'une nouvelle étude ne soit pas écrasée dans le même bucket.

    `n_jobs` parallélise les trials (threads). Comme un fit TabPFN est un appel réseau (I/O-bound),
    le GIL se libère pendant l'attente → gain quasi linéaire. Le sampler TPE est configuré en
    `constant_liar` (pénalise les trials en vol pour que les workers n'explorent pas la même région)
    et `multivariate` (modélise les corrélations de l'espace ~21D) — adapté à cette concurrence.
    """
    if (cutoffs is None) == (tournaments is None):
        raise ValueError("Fournir exactement l'un de `cutoffs` (calendaire) ou `tournaments` (LOTO).")
    experiment = experiment or f"{study_name}-optuna"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state, constant_liar=True, multivariate=True
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        load_if_exists=True,
    )

    if tournaments is not None:
        target = lambda trial: tournament_objective(
            trial,
            tournaments=tournaments,
            train_years_range=train_years_range,
            max_train=max_train,
            random_state=random_state,
            log_mlflow=log_mlflow,
            classifier_factory=classifier_factory,
            experiment=experiment,
        )
    else:
        target = lambda trial: objective(
            trial,
            cutoffs=cutoffs,
            train_years_range=train_years_range,
            max_train=max_train,
            categories=categories,
            random_state=random_state,
            log_mlflow=log_mlflow,
            classifier_factory=classifier_factory,
            experiment=experiment,
        )

    study.optimize(target, n_trials=n_trials, n_jobs=n_jobs)
    print(f"best log_loss = {study.best_value:.4f}")
    print(f"best params   = {study.best_params}")
    return study


def _parse_cutoffs(value: str) -> tuple[date, ...]:
    return tuple(date.fromisoformat(x.strip()) for x in value.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="trials en parallèle (threads ; fits TabPFN I/O-bound). Attention au quota/débit API.",
    )
    parser.add_argument("--n-cutoffs", type=int, default=1, help="nb de cutoffs récents du pool")
    parser.add_argument(
        "--cutoffs", type=_parse_cutoffs, default=None,
        help="liste explicite, ex. 2024-01-01,2025-01-01 (prioritaire sur --n-cutoffs)",
    )
    parser.add_argument(
        "--tournament", action="store_true",
        help="optimise le log-loss LOTO tournoi (CDM+Euro+Copa) au lieu du calendaire",
    )
    parser.add_argument(
        "--tournaments", type=parse_tournaments, default=None,
        help="liste 'Nom:Année,...' à optimiser (implique --tournament ; défaut : pool complet)",
    )
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument(
        "--experiment", default=None,
        help="nom de l'expérience MLflow (défaut : <study-name>-optuna)",
    )
    parser.add_argument("--storage", default=OPTUNA_STORAGE)
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le logging MLflow")
    args = parser.parse_args()

    # Métrique calendaire (défaut) ou LOTO tournoi (--tournament / --tournaments).
    tournament_mode = args.tournament or args.tournaments is not None
    run_study(
        n_trials=args.n_trials,
        cutoffs=None if tournament_mode else (args.cutoffs or default_cutoffs(args.n_cutoffs)),
        tournaments=(args.tournaments or DEFAULT_TOURNAMENTS) if tournament_mode else None,
        max_train=args.max_train,
        study_name=args.study_name,
        experiment=args.experiment,
        storage=args.storage,
        log_mlflow=not args.no_mlflow,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
