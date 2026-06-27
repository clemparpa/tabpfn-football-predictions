"""Feature ELO (rating FIFA-style).

L'ELO est récursif et globalement couplé (le résultat d'un match modifie le rating des
deux équipes, qui resservent dans leurs matchs suivants) : il n'est pas vectorisable. On
fait une unique passe chronologique sur un dict de ratings.

Anti-leakage : on enregistre `home_elo`/`away_elo` AVANT la mise à jour, de sorte que le
rating attribué à un match reflète uniquement les rencontres antérieures. Le rating est
enregistré pour TOUS les matchs (y compris non joués) : un fixture futur reçoit ainsi le
rating courant des équipes, indispensable pour prédire.

Contrat : `add_elo(res, cfg)` renvoie `[match_id, home_elo, away_elo]`, prêt à joindre.
"""
from collections import defaultdict

import polars as pl

from training.config import FeatureConfig


def _goal_diff_multiplier(goal_diff: int, cfg: FeatureConfig) -> float:
    """Multiplicateur FIFA : une victoire large déplace davantage les ratings."""
    margin = abs(goal_diff)
    if margin <= 1:
        return cfg.gd_mult_small
    if margin == 2:
        return cfg.gd_mult_medium
    return (cfg.gd_mult_intercept + margin) / cfg.gd_mult_divisor


def _k_factor(category, cfg: FeatureConfig) -> float:
    """K = k_base * multiplicateur de la catégorie (fallback really_minor si cat null)."""
    mult = cfg.k_mult_by_category
    if category is None or not (1 <= category <= len(mult)):
        return cfg.k_base * mult[-1]
    return cfg.k_base * mult[category - 1]


def add_elo(res: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Calcule l'ELO pré-match de chaque équipe et renvoie [match_id, home_elo, away_elo]."""
    ratings: dict[str, float] = defaultdict(lambda: cfg.elo_base)

    match_ids: list[int] = []
    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in res.sort("date", "match_id").iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        home_rating, away_rating = ratings[home], ratings[away]

        # rating pré-match enregistré pour tous les matchs (le futur reçoit le rating courant)
        match_ids.append(row["match_id"])
        home_elos.append(home_rating)
        away_elos.append(away_rating)

        if not row["finished"]:
            continue

        home_advantage = cfg.home_adv * (0.0 if row["neutral"] else 1.0)
        goal_diff = row["home_score"] - row["away_score"]
        expected = 1.0 / (
            1.0 + 10.0 ** ((away_rating - home_rating - home_advantage) / cfg.elo_scale)
        )
        score = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)
        k_factor = _k_factor(row["tournament_category"], cfg)
        delta = k_factor * _goal_diff_multiplier(goal_diff, cfg) * (score - expected)

        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta

    return pl.DataFrame(
        {
            "match_id": match_ids,
            "home_elo": home_elos,
            "away_elo": away_elos,
        },
        schema={"match_id": res.schema["match_id"], "home_elo": pl.Float64, "away_elo": pl.Float64},
    )
