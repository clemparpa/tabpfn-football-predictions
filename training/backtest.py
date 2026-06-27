"""Découpage train/test pour le backtesting, sans fuite temporelle.

L'idée : simuler une date « aujourd'hui » (`cutoff`). Tout match à partir de cette date est
masqué (passé en « non joué »), de sorte que le pipeline de features le traite exactement
comme un fixture à prédire — ELO/H2H/forme reportent la dernière valeur connue *avant* le
cutoff, sans regarder l'avenir. On entraîne ensuite sur une fenêtre d'historique
(`train_years` années avant le cutoff) et on teste sur les matchs réellement joués à partir
du cutoff.

Le module ne fait que le découpage des données (features + label `outcome` + identifiants) :
le choix et l'entraînement du modèle restent à l'appelant.

Garantie anti-fuite : valable uniquement pour un découpage *chronologique* (tout `>= cutoff`
masqué). Le report de forme de `team_form` lit le dernier match joué global de chaque équipe,
qui est alors forcément antérieur au cutoff — donc sans fuite. Ne pas réutiliser pour un
holdout dispersé.
"""
from typing import Literal
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date

import polars as pl

from training.config import FeatureConfig
from training.data import TOURNAMENT_CATEGORY_LABELS, load_matches
from training.features import build_features

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


def mask_after(res: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Passe en « non joué » tout match dont `date >= cutoff` (scores -> null, finished -> False).

    Les matchs antérieurs au cutoff sont laissés intacts. C'est la primitive anti-fuite : un
    match masqué devient un pur point de requête pour les features.
    """
    is_future = pl.col("date") >= cutoff
    return res.with_columns(
        home_score=pl.when(is_future).then(None).otherwise(pl.col("home_score")),
        away_score=pl.when(is_future).then(None).otherwise(pl.col("away_score")),
        finished=pl.col("finished") & ~is_future,
    )


def add_outcome(res: pl.DataFrame) -> pl.DataFrame:
    """Calcule le label `[match_id, outcome]` sur les matchs joués (scores d'origine).

    `home_win` / `away_win` / `draw` — miroir polars du `np.select` de `predict.py`. Les
    matchs non joués sont absents du frame renvoyé (donc `outcome` null après un left join).
    """
    return res.filter(pl.col("finished")).select(
        "match_id",
        outcome=pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(pl.lit("home_win"))
        .when(pl.col("home_score") < pl.col("away_score"))
        .then(pl.lit("away_win"))
        .otherwise(pl.lit("draw")),
    )


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
    - `categories` : labels de catégorie de tournoi à *conserver* (None = toutes). Les autres
      sont retirées du dataset **avant** le calcul des features (n'alimentent pas l'historique).
    - `res` : frame injectable (défaut `load_matches()`), pratique pour les tests.
    """
    categories = _validate_categories(categories)
    if res is None:
        res = load_matches()
    if categories is not None:
        res = res.filter(pl.col("tournament_category_label").is_in(list(categories)))

    labels = add_outcome(res)  # vrais résultats, avant masquage
    masked = mask_after(res, cutoff)
    wide = build_features(masked, cfg).join(labels, on="match_id", how="left")

    train_start = pl.select(pl.lit(cutoff).dt.offset_by(f"-{train_years}y")).item()
    played = pl.col("outcome").is_not_null()  # un match réellement joué (label disponible)
    train = wide.filter(
        (pl.col("date") >= train_start) & (pl.col("date") < cutoff) & played
    )
    test = wide.filter((pl.col("date") >= cutoff) & played)

    return BacktestSplit(
        train=train,
        test=test,
        cutoff=cutoff,
        train_start=train_start,
        categories=categories,
    )
