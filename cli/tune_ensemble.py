"""Tuning Optuna de la couche d'ensemble (GBM sur TabPFN), TabPFN figé.

Stratégie : on **fige** TabPFN + features + `train_years` depuis une run MLflow « source »
(produite par `training.tuning`) et on n'optimise plus que la couche d'ensemble — poids,
méthode de combinaison, et `C` de la régression logistique.

Gain clé : TabPFN figé ⇒ ses `predict_proba` sur le test du backtest sont **déterministes**. On
fitte donc TabPFN **une seule fois par cutoff** (seul appel réseau du script), on **cache** ses
probas, puis Optuna balaye la couche d'ensemble **100 % en local** — le GBM se refit à chaque
trial mais c'est rapide. Des centaines de trials ne coûtent ainsi aucun appel API supplémentaire.

La meilleure config est loggée dans MLflow comme une run **autonome** : on recopie les params figés
(`cfg.*`, `tabpfn.*`, `feature_columns`, `train_years`) et on ajoute le bloc `ensemble.*`, si bien
que `submit.py --run-id <run ensemble>` peut tout reconstruire depuis cette seule run.

Exemple (depuis la racine du repo) :
    python -m cli.tune_ensemble --source-run-id <run source> --n-cutoffs 1 --n-trials 50
"""
import argparse
from datetime import date

import numpy as np
import optuna
from sklearn.metrics import log_loss

from training.backtest import make_backtest_split
from training.ensemble import build_gbm, combine_probas
from training.mlflow_io import load_run_params, reconstruct_from_params
from training.model import DEFAULT_MAX_TRAIN, MLFLOW_TRACKING_URI, _feature_matrix
from training.tuning import OPTUNA_STORAGE, build_tabpfn, default_cutoffs

DEFAULT_STUDY_NAME = "tabpfn-football-ensemble"  # étude Optuna distincte de la base TabPFN
ENSEMBLE_EXPERIMENT = "tabpfn-football-ensemble"  # expérience MLflow dédiée
RANDOM_STATE = 42


def _align(proba: np.ndarray, est_classes, target_classes) -> np.ndarray:
    """Réordonne les colonnes de `proba` (ordre `est_classes`) sur `target_classes`."""
    order = [list(est_classes).index(c) for c in target_classes]
    return np.asarray(proba, dtype=np.float64)[:, order]


def build_cache(cfg, feature_columns, tabpfn_kwargs, train_years, cutoffs) -> list[dict]:
    """Pré-calcule, pour chaque cutoff, les probas TabPFN sur le test (un seul fit TabPFN / cutoff).

    Renvoie une liste de dicts `{Xtr, ytr, Xte, yte, classes, p_tabpfn}` réutilisés par tous les
    trials Optuna. C'est ici (et seulement ici) que TabPFN appelle l'API.
    """
    cache: list[dict] = []
    for cutoff in cutoffs:
        split = make_backtest_split(cutoff, train_years, None, cfg)
        train = split.train.sort("date").tail(DEFAULT_MAX_TRAIN)
        Xtr = _feature_matrix(train, feature_columns)
        ytr = train.get_column("outcome").to_numpy()
        Xte = _feature_matrix(split.test, feature_columns)
        yte = split.test.get_column("outcome").to_numpy()

        classes = np.unique(ytr)
        print(f"  cutoff {cutoff} : fit TabPFN (train {train.height}, test {split.test.height})…")
        clf = build_tabpfn(tabpfn_kwargs, RANDOM_STATE)
        clf.fit(Xtr, ytr)
        p_tabpfn = _align(clf.predict_proba(Xte), clf.classes_, classes)

        cache.append(
            dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, classes=classes, p_tabpfn=p_tabpfn)
        )
    return cache


def _mean_log_loss(cache, *, combine_probas_fn) -> float:
    """Moyenne du log-loss sur les cutoffs cachés, en appliquant `combine_probas_fn(item)`."""
    losses = []
    for item in cache:
        proba = combine_probas_fn(item)
        losses.append(log_loss(item["yte"], proba, labels=item["classes"]))
    return float(np.mean(losses))


def tabpfn_baseline(cache) -> float:
    """Log-loss de référence : TabPFN seul (clip/renorm via combine_probas, weight=1)."""
    return _mean_log_loss(
        cache,
        combine_probas_fn=lambda it: combine_probas(
            it["p_tabpfn"], it["p_tabpfn"], method="arith", weight=1.0
        ),
    )


def make_objective(cache):
    """Construit l'objectif Optuna fermé sur le cache (aucun appel réseau)."""

    def objective(trial) -> float:
        weight = trial.suggest_float("weight", 0.0, 1.0)
        method = trial.suggest_categorical("combine", ["arith", "geom"])
        gbm_kwargs = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            max_iter=trial.suggest_int("max_iter", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            l2_regularization=trial.suggest_float("l2_regularization", 0.0, 10.0),
        )

        def combine_for(item):
            gbm = build_gbm(gbm_kwargs, RANDOM_STATE).fit(item["Xtr"], item["ytr"])
            p_gbm = _align(gbm.predict_proba(item["Xte"]), gbm.classes_, item["classes"])
            return combine_probas(item["p_tabpfn"], p_gbm, method=method, weight=weight)

        return _mean_log_loss(cache, combine_probas_fn=combine_for)

    return objective


def log_best_to_mlflow(
    source_params: dict, source_run_id: str, best_params: dict, best_loss: float, experiment: str
) -> None:
    """Logge la meilleure config d'ensemble en run MLflow **autonome**.

    Recopie les params figés de la source (cfg/tabpfn/feature_columns/train_years) et ajoute le
    bloc `ensemble.*` + la provenance, pour que `submit.py` reconstruise tout depuis cette run.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="ensemble"):
        params = dict(source_params)  # recopie figée (cfg.*, tabpfn.*, feature_columns, train_years)
        params.update(
            {
                "ensemble.use_gbm": True,
                "ensemble.combine": best_params["combine"],
                "ensemble.weight": best_params["weight"],
                "source_run_id": source_run_id,
            }
        )
        # Hyperparams GBM : bloc `gbm.*` (miroir de `tabpfn.*`).
        for key in ("learning_rate", "max_iter", "max_depth", "l2_regularization"):
            params[f"gbm.{key}"] = best_params[key]
        mlflow.log_params(params)
        mlflow.log_metric("log_loss", best_loss)


def _parse_cutoffs(value: str) -> tuple[date, ...]:
    return tuple(date.fromisoformat(x.strip()) for x in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-run-id", default=None, help="run MLflow source à figer (prioritaire)"
    )
    parser.add_argument(
        "--source-experiment", default=None,
        help="à défaut de --source-run-id : meilleure run (log-loss) de cet experiment",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-cutoffs", type=int, default=1, help="nb de cutoffs récents du pool")
    parser.add_argument(
        "--cutoffs", type=_parse_cutoffs, default=None,
        help="liste explicite, ex. 2024-01-01,2025-01-01 (prioritaire sur --n-cutoffs)",
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", default=OPTUNA_STORAGE)
    parser.add_argument("--experiment", default=ENSEMBLE_EXPERIMENT, help="experiment MLflow de sortie")
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le logging MLflow")
    args = parser.parse_args()

    if not (args.source_run_id or args.source_experiment):
        parser.error("fournir --source-run-id ou --source-experiment (la run TabPFN à figer)")

    # 1. Charge et fige la source.
    source_params, source_run_id, source_ll = load_run_params(
        args.source_run_id, args.source_experiment
    )
    cfg, feature_columns, tabpfn_kwargs, train_years = reconstruct_from_params(source_params)
    ll = f"{source_ll:.4f}" if source_ll is not None else "?"
    print(f"Source : run {source_run_id} (log_loss={ll})")
    print(f"  train_years={train_years} | {len(feature_columns)} features | tabpfn={tabpfn_kwargs}")

    # 2. Cache des probas TabPFN (seul appel réseau).
    cutoffs = args.cutoffs or default_cutoffs(args.n_cutoffs)
    print(f"Caching TabPFN sur {len(cutoffs)} cutoff(s)…")
    cache = build_cache(cfg, feature_columns, tabpfn_kwargs, train_years, cutoffs)
    baseline = tabpfn_baseline(cache)
    print(f"Baseline TabPFN seul : log_loss = {baseline:.4f}")

    # 3. Étude Optuna locale.
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )
    study.optimize(make_objective(cache), n_trials=args.n_trials)

    best = study.best_value
    delta = baseline - best
    print(f"\nBest ensemble log_loss = {best:.4f}  (baseline {baseline:.4f}, Δ {delta:+.4f})")
    print(f"Best params = {study.best_params}")
    if delta <= 0:
        print("⚠️  L'ensemble n'améliore pas TabPFN seul sur ce(s) cutoff(s).")

    # 4. Run MLflow autonome.
    if not args.no_mlflow:
        log_best_to_mlflow(
            source_params, source_run_id, study.best_params, best, args.experiment
        )
        print(f"Run d'ensemble loggée dans l'experiment {args.experiment!r}.")


if __name__ == "__main__":
    main()
