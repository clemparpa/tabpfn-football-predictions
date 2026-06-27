"""Orchestration des familles de features.

`build_features` joint chaque famille (contrat `match_id -> home_*/away_*`) sur `res`,
puis calcule les diffs (qui vivent ici, après les joins, pour qu'aucun module n'ait
besoin de connaître les colonnes d'un autre).
"""
import polars as pl

from training.config import FeatureConfig
from training.features.elo import add_elo
from training.features.h2h import add_h2h
from training.features.team_form import add_team_form


def build_features(res: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Construit le frame wide des features (1 ligne par match)."""
    return (
        res
        .join(add_team_form(res, cfg), on="match_id", how="left")
        .join(add_elo(res, cfg), on="match_id", how="left")
        .join(add_h2h(res, cfg), on="match_id", how="left")
        .with_columns(
            points_history_diff=pl.col("home_points_history") - pl.col("away_points_history"),
            goal_diff_history_diff=pl.col("home_goal_diff_history") - pl.col("away_goal_diff_history"),
            # elo_diff intègre l'avantage du terrain (nul si match sur terrain neutre)
            elo_diff=(
                pl.col("home_elo")
                + cfg.home_adv * (1 - pl.col("neutral").cast(pl.Int8))
                - pl.col("away_elo")
            ),
        )
    )
