"""Génère le CSV de soumission du concours avec le pipeline tuné.

Les hyperparamètres peuvent venir de deux sources :

1. **MLflow** (recommandé) : on lit une run loggée par le tuning (`--experiment` pour prendre
   la meilleure run par log-loss, ou `--run-id` pour en cibler une précise). On reconstruit
   alors `FeatureConfig`, la liste de colonnes (`feature_columns`) et les kwargs TabPFN
   directement depuis les params loggés — aucune recopie manuelle.
2. **Constantes en dur** (fallback, si aucun flag MLflow) : le bloc `WINNING_*` ci-dessous.

On réentraîne ensuite `TabPFNClassifier` sur l'historique récent et on prédit les matchs non
joués (`finished=False`) de `results.csv`.

Contrairement à la baseline `predict.py` (pandas, isolée), ce script réutilise le pipeline
polars du package `training/` : `load_matches` → `build_features` → `train_classifier`. Les
features des matchs futurs sont déjà gérées par les builders (`team_form` reporte la dernière
forme connue, `elo`/`h2h` enregistrent l'état pré-match).

Exemples (depuis la racine du repo) :
    python submit.py                                   # constantes en dur
    python submit.py --experiment lean-tune-optuna     # meilleure run de l'experiment
    python submit.py --run-id 5191988f39ec443cb55a07ea80c28084
"""
import argparse
from datetime import date, timedelta

import numpy as np
import polars as pl

from training.backtest import add_outcome, make_backtest_split
from training.config import FeatureConfig
from training.data import load_matches
from training.ensemble import EnsembleConfig, build_model
from training.features import build_features
from training.mlflow_io import (
    load_run_params,
    reconstruct_ensemble_config,
    reconstruct_from_params,
)
from training.model import (
    DEFAULT_MAX_TRAIN,
    FEATURE_GROUPS,
    _feature_matrix,
    evaluate,
    train_classifier,
)
from training.proba import clip_renorm

RANDOM_STATE = 42

# --- Fallback : hyperparamètres en dur (utilisés si aucun flag MLflow) -------------------
# Meilleur run lean MLflow (log-loss ≈ 0.83 : team_form + diffs + context, history_size=9).
WINNING_TRAIN_YEARS = 10
WINNING_GROUPS = ("team_form", "diffs", "context")
WINNING_COLUMNS = tuple(c for g in WINNING_GROUPS for c in FEATURE_GROUPS[g])
WINNING_CFG = FeatureConfig(
    points_history_size=9,
    winrate_history_size=9,
    drawrate_history_size=9,
    scores_history_size=9,
    goal_diff_history_size=9,
    home_adv=75.28507319896863,
    k_base=44.01341183459286,
    elo_scale=442.50335890278603,
)
WINNING_TABPFN_KWARGS = dict(n_estimators=8)


# --- Résolution de la config (MLflow ou constantes en dur) -------------------------------

def resolve_config(args) -> tuple[FeatureConfig, tuple[str, ...], EnsembleConfig, int, str]:
    """Détermine la source des hyperparamètres (MLflow si flag, sinon constantes en dur).

    Renvoie `(cfg_features, feature_columns, ensemble_cfg, train_years, source)`. L'`EnsembleConfig`
    porte les kwargs TabPFN **et** la couche d'ensemble (`ensemble.*` si la run en contient).
    """
    if args.run_id or args.experiment:
        params, run_id, log_loss = load_run_params(args.run_id, args.experiment)
        cfg, columns, tabpfn_kwargs, train_years = reconstruct_from_params(params)
        ensemble_cfg = reconstruct_ensemble_config(params, tabpfn_kwargs)
        ll = f"{log_loss:.4f}" if log_loss is not None else "?"
        source = f"MLflow run {run_id} (log_loss={ll})"
    else:
        cfg, columns, train_years = WINNING_CFG, WINNING_COLUMNS, WINNING_TRAIN_YEARS
        ensemble_cfg = EnsembleConfig(tabpfn_kwargs=dict(WINNING_TABPFN_KWARGS))
        source = "constantes en dur (WINNING_*)"
    return cfg, columns, ensemble_cfg, train_years, source


def apply_tabpfn_overrides(kwargs: dict, *, n_estimators, thinking, thinking_effort) -> dict:
    """Applique les overrides CLI sur les kwargs TabPFN issus de la source (MLflow ou en dur).

    `n_estimators`/`thinking` à None laissent la valeur de la source intacte ; sinon ils la
    remplacent. `thinking=True` ajoute `thinking_mode`+`thinking_effort` ; `thinking=False` les retire.
    """
    kwargs = dict(kwargs)
    if n_estimators is not None:
        kwargs["n_estimators"] = n_estimators
    if thinking is True:
        kwargs["thinking_mode"] = True
        kwargs["thinking_effort"] = thinking_effort
    elif thinking is False:
        kwargs.pop("thinking_mode", None)
        kwargs.pop("thinking_effort", None)
    return kwargs


# --- Pipeline de soumission --------------------------------------------------------------

def backtest_sanity_check(cfg, feature_columns, ensemble_cfg, train_years) -> None:
    """Réentraîne/évalue sur le cutoff 2025-01-01 pour valider le pipeline tuné."""
    split = make_backtest_split(date(2025, 1, 1), train_years, None, cfg)
    train = split.train.sort("date").tail(DEFAULT_MAX_TRAIN)
    clf = build_model(ensemble_cfg, RANDOM_STATE)
    clf = train_classifier(train, clf, RANDOM_STATE, feature_columns)
    metrics = evaluate(clf, split.test, feature_columns)
    print(
        f"Backtest {split.cutoff} ({metrics['n_test']} matchs) : "
        f"accuracy {metrics['accuracy']:.0%}, log-loss {metrics['log_loss']:.3f} "
        f"(train {train.height})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default=None, help="charge la meilleure run (log-loss) de cet experiment MLflow")
    parser.add_argument("--run-id", default=None, help="charge une run MLflow précise (prioritaire sur --experiment)")
    parser.add_argument("--n-estimators", type=int, default=None, help="override n_estimators TabPFN (défaut : valeur de la source)")
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=None,
        help="force le mode thinking : --thinking l'active, --no-thinking le désactive (défaut : valeur de la source)",
    )
    parser.add_argument("--thinking-effort", choices=["medium", "high"], default="medium", help="effort du mode thinking (si activé)")
    parser.add_argument("--no-gbm", action="store_true", help="désactive la couche GBM (TabPFN seul) même si la run l'active")
    parser.add_argument("--weight", type=float, default=None, help="override du poids TabPFN dans l'ensemble (0..1)")
    parser.add_argument("--out", default=None, help="chemin du CSV de sortie (défaut : predictions_YYYYMMDD.csv)")
    parser.add_argument("--no-backtest", action="store_true", help="saute le backtest de contrôle")
    args = parser.parse_args()

    cfg, feature_columns, ensemble_cfg, train_years, source = resolve_config(args)
    ensemble_cfg.tabpfn_kwargs = apply_tabpfn_overrides(
        ensemble_cfg.tabpfn_kwargs,
        n_estimators=args.n_estimators,
        thinking=args.thinking,
        thinking_effort=args.thinking_effort,
    )
    if args.no_gbm:
        ensemble_cfg.use_gbm = False
    if args.weight is not None:
        ensemble_cfg.weight = args.weight
    print(f"Source des params : {source}")
    print(f"train_years={train_years} | {len(feature_columns)} features | tabpfn={ensemble_cfg.tabpfn_kwargs}")
    if ensemble_cfg.use_gbm:
        print(f"Ensemble : GBM activé (combine={ensemble_cfg.combine}, weight={ensemble_cfg.weight:.3f}, gbm={ensemble_cfg.gbm_kwargs})")
    else:
        print("Ensemble : TabPFN seul (GBM désactivé)")

    res = load_matches()
    latest = res.filter(pl.col("finished")).get_column("date").max()
    print(f"Dernier match joué : {latest}")

    # Un seul passage de features : identiques pour le train et les fixtures à prédire.
    wide = build_features(res, cfg).join(add_outcome(res), on="match_id", how="left")

    # Entraînement : matchs joués des `train_years` dernières années, plafonnés (limite API).
    cutoff = latest + timedelta(days=1)
    train_start = pl.select(pl.lit(cutoff).dt.offset_by(f"-{train_years}y")).item()
    train = (
        wide.filter(
            pl.col("outcome").is_not_null()
            & (pl.col("date") >= train_start)
            & (pl.col("date") < cutoff)
        )
        .sort("date")
        .tail(DEFAULT_MAX_TRAIN)
    )
    future = wide.filter(~pl.col("finished")).sort("date")
    print(f"Train : {train.height} matchs ({train_start} → {cutoff}) | fixtures : {future.height}")

    if not args.no_backtest:
        backtest_sanity_check(cfg, feature_columns, ensemble_cfg, train_years)

    # Fit final + prédiction des fixtures.
    clf = build_model(ensemble_cfg, RANDOM_STATE)
    clf = train_classifier(train, clf, RANDOM_STATE, feature_columns)
    proba = clf.predict_proba(_feature_matrix(future, feature_columns))

    # Clip + renormalisation : garantit 0 < p < 1 et somme ≈ 1 (rules.md §5).
    proba = clip_renorm(proba)
    classes = list(clf.classes_)
    col = {c: proba[:, i] for i, c in enumerate(classes)}
    predicted = np.array(classes)[proba.argmax(axis=1)]

    # Schéma de soumission officiel : pas de colonne `predicted` (cf. sample du concours).
    out = future.select("date", "home_team", "away_team").with_columns(
        p_home_win=pl.Series(col["home_win"]),
        p_draw=pl.Series(col["draw"]),
        p_away_win=pl.Series(col["away_win"]),
    )

    today_str = date.today().strftime("%Y%m%d")
    filename = args.out or f"predictions_{today_str}.csv"
    out.write_csv(filename)
    print(f"\n{out.height} prédictions → {filename}\n")
    for r, pred in zip(out.iter_rows(named=True), predicted):
        print(
            f"  {r['date']}  {r['home_team']:>20} vs {r['away_team']:<20}  "
            f"-> {pred:<9}  "
            f"H {r['p_home_win']:4.0%} | D {r['p_draw']:4.0%} | A {r['p_away_win']:4.0%}"
        )


if __name__ == "__main__":
    main()
