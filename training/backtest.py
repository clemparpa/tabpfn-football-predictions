"""Découpage train/test pour le backtesting, sans fuite temporelle.

L'idée : simuler une date « aujourd'hui » (`cutoff`). On entraîne sur une fenêtre
d'historique (`train_years` années avant le cutoff) et on teste sur les matchs réellement
joués à partir du cutoff.

Les features sont calculées sur les données **réelles** (non masquées), exactement comme la
baseline `predict.py` : chaque match du test reçoit donc l'état (ELO/forme/H2H) reflétant
*tous* les matchs joués avant son coup d'envoi — y compris les autres matchs du test joués
plus tôt dans la fenêtre. C'est l'évaluation « au fil de l'eau » (rolling), comparable 1:1
au backtest de la baseline.

Garantie anti-fuite : elle ne vient PAS d'un masquage, mais des feature builders eux-mêmes,
qui calculent du strictement pré-match pour tout match `finished` (`team_form` via `shift(1)`,
`elo` en enregistrant le rating avant mise à jour, `h2h` en retranchant la contribution
propre du match). Un match joué n'utilise donc jamais son propre résultat ni un match futur.

Le module ne fait que le découpage des données (features + label `outcome` + identifiants) :
le choix et l'entraînement du modèle restent à l'appelant.
"""
from typing import Literal
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date

import polars as pl

from training.config import FeatureConfig
from training.data import TOURNAMENT_CATEGORY_LABELS, load_matches
from training.evaluation import add_outcome, build_eval_frame, select_rows

__all__ = ["BacktestSplit", "add_outcome", "make_backtest_split"]

type TournamentCategory = Literal[
    "world",
    "continental_major",
    "qualification_and_nations_leagues",
    "regional",
    "minor",
    "non_fifa",
    "really_minor"
]



@dataclass(frozen=True)
class BacktestSplit:
    """Résultat d'un découpage de backtest.

    `train` / `test` sont des frames complets (features `home_*`/`away_*`, diffs, label
    `outcome`, et identifiants `match_id`/`date`/`home_team`/`away_team`). `cutoff`,
    `train_start` et `categories` rappellent les bornes utilisées.
    """

    train: pl.DataFrame
    test: pl.DataFrame
    cutoff: date
    train_start: date
    categories: tuple[str, ...] | None


def _validate_categories(categories: Collection[TournamentCategory] | None) -> tuple[TournamentCategory, ...] | None:
    """Vérifie que les labels demandés existent ; renvoie un tuple figé (ou None)."""
    if categories is None:
        return None
    requested = tuple(categories)
    unknown = [c for c in requested if c not in TOURNAMENT_CATEGORY_LABELS]
    if unknown:
        raise ValueError(
            f"Catégorie(s) de tournoi inconnue(s) : {unknown}. "
            f"Labels valides : {list(TOURNAMENT_CATEGORY_LABELS)}"
        )
    return requested


def make_backtest_split(
    cutoff: date,
    train_years: int,
    categories: Collection[TournamentCategory] | None = None,
    cfg: FeatureConfig = FeatureConfig(),
    res: pl.DataFrame | None = None,
) -> BacktestSplit:
    """Construit le découpage train/test au `cutoff` donné.

    - `cutoff` : tout match `>= cutoff` est masqué (= « futur » à prédire).
    - `train_years` : profondeur d'historique pour l'entraînement, `[cutoff - train_years, cutoff)`.
    - `categories` : labels de catégorie de tournoi à *conserver* (None = toutes). ⚠️ Filtre **amont** :
      les autres sont retirées du dataset **avant** le calcul des features (n'alimentent pas
      l'historique). Pour restreindre seulement les *lignes* évaluées sans toucher aux features,
      passer par la couture `training.evaluation` (`build_eval_frame` + `select_rows`).
    - `res` : frame injectable (défaut `load_matches()`), pratique pour les tests.

    Mince wrapper au-dessus de la couture `training.evaluation` : `build_eval_frame` calcule les
    features sur tout le dataset (rolling, anti-fuite portée par les builders), `select_rows`
    découpe les fenêtres train/test.
    """
    categories = _validate_categories(categories)
    if res is None:
        res = load_matches()
    if categories is not None:
        res = res.filter(pl.col("tournament_category_label").is_in(list(categories)))

    wide = build_eval_frame(cfg, res)

    train_start = pl.select(pl.lit(cutoff).dt.offset_by(f"-{train_years}y")).item()
    played = pl.col("outcome").is_not_null()  # un match réellement joué (label disponible)
    train, test = select_rows(
        wide,
        train_pred=(pl.col("date") >= train_start) & (pl.col("date") < cutoff) & played,
        test_pred=(pl.col("date") >= cutoff) & played,
    )

    return BacktestSplit(
        train=train,
        test=test,
        cutoff=cutoff,
        train_start=train_start,
        categories=categories,
    )
