"""Backtest « tournoi » : évaluer le modèle sur de vrais matchs de grand tournoi.

Le backtest calendaire (`training.backtest`) teste sur *tous* les matchs ≥ une date — dominé par
des amicaux/qualifs à domicile, loin de la cible (CDM 2026, terrain neutre). Ici on teste sur les
matchs **du tournoi lui-même** : pour chaque édition, on entraîne sur tout l'historique antérieur
et on évalue uniquement ses matchs, puis on **agrège en leave-one-tournament-out (LOTO)** — la
moyenne des log-loss par tournoi, pour ne pas sur-régler sur une seule édition.

Tout repose sur la couture `training.evaluation` :

    build_eval_frame(cfg, res)   # features sur TOUT le dataset, une fois (anti-fuite via builders)
        → select_rows(...)       # filtre de LIGNES : train = pré-tournoi, test = matchs du tournoi
        → predict_proba_frame    # probas valides + alignées
        → score                  # log-loss + accuracy

Filtrer les lignes ne touche pas aux features (cf. piège n°1 du backlog) : l'ELO d'un match de
tournoi reflète tout l'historique réel d'avant son coup d'envoi, pas un historique amputé.

Le `fit_fn` d'`evaluate_tournaments` est **injectable** : un faux classifieur sklearn permet de
tester toute l'orchestration **hors-ligne**. Le défaut (`train_classifier`) entraîne un vrai
TabPFN distant — un fit par tournoi, sur quota d'API : ne le déclencher qu'avec accord explicite.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from training.config import FeatureConfig
from training.evaluation import build_eval_frame, predict_proba_frame, score, select_rows
from training.model import DEFAULT_MAX_TRAIN, FEATURE_COLUMNS


@dataclass(frozen=True)
class Tournament:
    """Une édition de tournoi, ciblée par son nom (colonne `tournament`) et son année calendaire.

    `name` est le libellé exact de la source (ex. `"FIFA World Cup"`, `"UEFA Euro"`,
    `"Copa América"`) ; `year` l'année où l'édition s'est **jouée** (Euro 2020 → 2021, COVID).
    """

    name: str
    year: int

    @property
    def label(self) -> str:
        return f"{self.name} {self.year}"

    @property
    def row_pred(self) -> pl.Expr:
        """Prédicat polars sélectionnant les lignes de cette édition."""
        return (pl.col("tournament") == self.name) & (pl.col("date").dt.year() == self.year)


# Pool par défaut : CDM (cœur de cible) + Euro + Copa América pour épaissir le LOTO. Éditions
# choisies pour leur volume et un historique d'entraînement suffisant en amont.
WORLD_CUPS: tuple[Tournament, ...] = tuple(Tournament("FIFA World Cup", y) for y in (2014, 2018, 2022))
EUROS: tuple[Tournament, ...] = tuple(Tournament("UEFA Euro", y) for y in (2016, 2021, 2024))
COPA_AMERICAS: tuple[Tournament, ...] = tuple(Tournament("Copa América", y) for y in (2019, 2021, 2024))
DEFAULT_TOURNAMENTS: tuple[Tournament, ...] = WORLD_CUPS + EUROS + COPA_AMERICAS


@dataclass(frozen=True)
class TournamentSplit:
    """Découpage train/test pour une édition : `cutoff` = veille du 1er match du tournoi."""

    tournament: Tournament
    cutoff: date
    train: pl.DataFrame
    test: pl.DataFrame


# Signature d'un entraîneur injectable : `(train_df, feature_columns) -> classifieur fitté`.
FitFn = Callable[[pl.DataFrame, Sequence[str]], object]


def tournament_split(
    tournament: Tournament,
    *,
    cfg: FeatureConfig = FeatureConfig(),
    res: pl.DataFrame | None = None,
    wide: pl.DataFrame | None = None,
    train_years: int | None = None,
) -> TournamentSplit:
    """Découpe le dataset pour une édition : train = pré-tournoi, test = matchs du tournoi.

    `wide` (frame de features pré-calculé via `build_eval_frame`) est injectable pour le
    réutiliser sur plusieurs tournois sans recalculer les features. À défaut, il est construit ici.
    Le `cutoff` est déduit des données (1er match joué de l'édition) — aucune date en dur.
    `train_years` borne optionnellement l'historique d'entraînement (défaut : tout l'historique).
    """
    if wide is None:
        wide = build_eval_frame(cfg, res)
    played = pl.col("outcome").is_not_null()  # match réellement joué (label disponible)
    test_pred = tournament.row_pred & played

    cutoff = wide.filter(test_pred).get_column("date").min()
    if cutoff is None:
        raise ValueError(f"Aucun match joué trouvé pour {tournament.label}.")

    train_pred = played & (pl.col("date") < cutoff)
    if train_years is not None:
        train_start = pl.select(pl.lit(cutoff).dt.offset_by(f"-{train_years}y")).item()
        train_pred = train_pred & (pl.col("date") >= train_start)

    train, test = select_rows(wide, train_pred=train_pred, test_pred=test_pred)
    return TournamentSplit(tournament=tournament, cutoff=cutoff, train=train, test=test)


@dataclass(frozen=True)
class TournamentReport:
    """Résultats d'un backtest LOTO : métriques par tournoi + moyennes agrégées."""

    per_tournament: tuple[dict, ...]
    loto_log_loss: float
    loto_accuracy: float


def _default_fit(train: pl.DataFrame, feature_columns: Sequence[str], *, random_state: int = 42):
    """Entraîneur par défaut : un vrai TabPFN (appel API). Import paresseux pour rester hors-ligne."""
    from training.model import train_classifier

    return train_classifier(train, None, random_state, feature_columns)


def evaluate_tournaments(
    tournaments: Sequence[Tournament] = DEFAULT_TOURNAMENTS,
    *,
    fit_fn: FitFn | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    cfg: FeatureConfig = FeatureConfig(),
    res: pl.DataFrame | None = None,
    max_train: int | None = DEFAULT_MAX_TRAIN,
) -> TournamentReport:
    """Backtest LOTO : un fit par tournoi, log-loss/accuracy par édition + moyenne.

    Les features sont calculées **une seule fois** (`build_eval_frame`) et réutilisées pour tous
    les tournois. `fit_fn(train_df, feature_columns) -> clf` est injectable : un faux classifieur
    rend l'orchestration testable hors-ligne. Le défaut entraîne un TabPFN réel (un fit par
    tournoi, sur quota) — à ne lancer qu'avec accord explicite.

    `max_train` plafonne l'entraînement aux lignes les plus récentes (limite de l'API TabPFN).
    """
    fit = fit_fn or _default_fit
    columns = tuple(feature_columns)
    wide = build_eval_frame(cfg, res)

    per_tournament: list[dict] = []
    for t in tournaments:
        split = tournament_split(t, wide=wide)
        train = split.train
        if max_train is not None and train.height > max_train:
            train = train.sort("date").tail(max_train)
        clf = fit(train, columns)
        metrics = score(predict_proba_frame(clf, split.test, columns))
        per_tournament.append(
            {
                "tournament": t.label,
                "cutoff": split.cutoff,
                "n_train": train.height,
                "n_test": metrics["n_test"],
                "log_loss": metrics["log_loss"],
                "accuracy": metrics["accuracy"],
            }
        )

    n = len(per_tournament)
    loto_log_loss = sum(r["log_loss"] for r in per_tournament) / n
    loto_accuracy = sum(r["accuracy"] for r in per_tournament) / n
    return TournamentReport(
        per_tournament=tuple(per_tournament),
        loto_log_loss=loto_log_loss,
        loto_accuracy=loto_accuracy,
    )
