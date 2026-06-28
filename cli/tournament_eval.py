"""Backtest « tournoi » en CLI : log-loss par grand tournoi + moyenne LOTO.

Évalue la config (MLflow ou constantes en dur, **même résolution que `cli.submit`**) sur de vrais
matchs de tournoi (`training.tournament`) : pour chaque édition, train sur tout l'historique
antérieur, test sur ses matchs, puis moyenne **leave-one-tournament-out**. C'est la métrique qui
ressemble à la cible (CDM 2026, neutre) — à préférer au log-loss calendaire global pour arbitrer.

⚠️ **Coût API** : chaque tournoi = un fit TabPFN distant (sur quota). Deux garde-fous :

1. `--dry-run` (faux classifieur local, **zéro réseau**) valide tout le plumbing hors-ligne.
2. **Cache disque des probas** : la frame de probas de chaque (config, tournoi) est écrite en
   parquet. Un re-run réutilise le cache et ne refait **aucun** fit (sauf `--refresh-cache`).

Exemples (depuis la racine du repo) :
    python -m cli.tournament_eval --dry-run                      # plumbing hors-ligne
    python -m cli.tournament_eval --experiment lean-tune-optuna  # vrais fits (gated par le cache)
    python -m cli.tournament_eval --tournaments "FIFA World Cup:2018,UEFA Euro:2021"
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl
from sklearn.dummy import DummyClassifier

from cli.submit import apply_tabpfn_overrides, resolve_config
from training.ensemble import build_model
from training.evaluation import build_eval_frame, feature_matrix, predict_proba_frame, score
from training.model import DEFAULT_MAX_TRAIN, train_classifier
from training.tournament import DEFAULT_TOURNAMENTS, parse_tournaments, tournament_split

RANDOM_STATE = 42
DEFAULT_CACHE_DIR = ".tournament_cache"


def config_key(cfg, feature_columns, ensemble_cfg, max_train) -> str:
    """Hash court et stable de la config — clé de cache (évite de mélanger deux configs)."""
    payload = json.dumps(
        {
            "cfg": asdict(cfg),
            "columns": list(feature_columns),
            "tabpfn": ensemble_cfg.tabpfn_kwargs,
            "use_gbm": ensemble_cfg.use_gbm,
            "combine": ensemble_cfg.combine,
            "weight": ensemble_cfg.weight,
            "gbm": ensemble_cfg.gbm_kwargs,
            "max_train": max_train,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def make_fit_fn(ensemble_cfg, *, dry_run: bool):
    """Construit le `fit_fn` : faux classifieur local (`dry_run`) ou vrai TabPFN distant."""
    if dry_run:
        def fit(train, feature_columns):
            clf = DummyClassifier(strategy="prior")
            clf.fit(feature_matrix(train, feature_columns), train.get_column("outcome").to_numpy())
            return clf
        return fit

    def fit(train, feature_columns):
        clf = build_model(ensemble_cfg, RANDOM_STATE)
        return train_classifier(train, clf, RANDOM_STATE, feature_columns)
    return fit


def proba_frame_for(
    tournament: Tournament, *, wide, fit_fn, feature_columns, max_train, cache_path: Path | None
) -> pl.DataFrame:
    """Renvoie la frame de probas du tournoi, depuis le cache si présent, sinon fit + écrit le cache."""
    if cache_path is not None and cache_path.exists():
        return pl.read_parquet(cache_path)

    split = tournament_split(tournament, wide=wide)
    train = split.train
    if max_train is not None and train.height > max_train:
        train = train.sort("date").tail(max_train)
    clf = fit_fn(train, tuple(feature_columns))
    frame = predict_proba_frame(clf, split.test, feature_columns)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(cache_path)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default=None, help="meilleure run (log-loss) de cet experiment MLflow")
    parser.add_argument("--run-id", default=None, help="run MLflow précise (prioritaire sur --experiment)")
    parser.add_argument("--n-estimators", type=int, default=None, help="override n_estimators TabPFN")
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=None,
        help="force le mode thinking (--thinking / --no-thinking ; défaut : valeur de la source)",
    )
    parser.add_argument("--thinking-effort", choices=["medium", "high"], default="medium")
    parser.add_argument(
        "--tournaments", type=parse_tournaments, default=DEFAULT_TOURNAMENTS,
        help="liste 'Nom:Année,...' (défaut : CDM + Euro + Copa)",
    )
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true", help="ne lit ni n'écrit le cache de probas")
    parser.add_argument("--refresh-cache", action="store_true", help="ignore le cache existant et le réécrit")
    parser.add_argument("--dry-run", action="store_true", help="faux classifieur local (zéro appel API)")
    args = parser.parse_args()

    cfg, feature_columns, ensemble_cfg, _train_years, source = resolve_config(args)
    ensemble_cfg.tabpfn_kwargs = apply_tabpfn_overrides(
        ensemble_cfg.tabpfn_kwargs,
        n_estimators=args.n_estimators,
        thinking=args.thinking,
        thinking_effort=args.thinking_effort,
    )
    # Ensembling GBM désactivé : on se recentre sur TabPFN seul (la couche GBM n'apportait rien).
    # On force TabPFN-only même si la run MLflow chargée portait un bloc `ensemble.*`.
    ensemble_cfg.use_gbm = False

    mode = "DRY-RUN (faux classifieur, zéro API)" if args.dry_run else "fits TabPFN réels"
    print(f"Source des params : {source}")
    print(f"Mode : {mode} | {len(feature_columns)} features | tabpfn={ensemble_cfg.tabpfn_kwargs}")
    print(f"Tournois : {len(args.tournaments)} édition(s)\n")

    fit_fn = make_fit_fn(ensemble_cfg, dry_run=args.dry_run)
    wide = build_eval_frame(cfg)

    # Le cache est désactivé en dry-run (probas factices) et par --no-cache.
    use_cache = not args.no_cache and not args.dry_run
    key = config_key(cfg, feature_columns, ensemble_cfg, args.max_train)
    cache_root = Path(args.cache_dir) / key

    rows: list[dict] = []
    for t in args.tournaments:
        cache_path = None
        if use_cache:
            cache_path = cache_root / f"{t.label.replace(' ', '_')}.parquet"
            if args.refresh_cache and cache_path.exists():
                cache_path.unlink()
        hit = cache_path is not None and cache_path.exists()  # cache présent AVANT l'appel
        frame = proba_frame_for(
            t, wide=wide, fit_fn=fit_fn, feature_columns=feature_columns,
            max_train=args.max_train, cache_path=cache_path,
        )
        metrics = score(frame)
        cached = "cache" if hit else "fit "
        rows.append({"tournament": t.label, **metrics})
        print(
            f"  [{cached}] {t.label:<22} ({metrics['n_test']:>2} matchs) : "
            f"log-loss {metrics['log_loss']:.3f} | accuracy {metrics['accuracy']:.0%}"
        )

    n = len(rows)
    loto_ll = sum(r["log_loss"] for r in rows) / n
    loto_acc = sum(r["accuracy"] for r in rows) / n
    print(f"\nLOTO ({n} tournois) : log-loss {loto_ll:.4f} | accuracy {loto_acc:.1%}")


if __name__ == "__main__":
    main()
