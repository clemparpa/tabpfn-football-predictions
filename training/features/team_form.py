"""Features de forme par équipe (rolling sur l'historique récent).

Toutes les features respectent la propriété anti-leakage : un `shift(1)` est appliqué
avant chaque `rolling_mean`, de sorte que seules les rencontres *antérieures* au coup
d'envoi entrent dans la fenêtre.

Contrat : `add_team_form(res, cfg)` renvoie un frame indexé par `match_id` avec les
colonnes `home_*` / `away_*`, prêt à être joint sur `res`.
"""
import polars as pl

from training.config import FeatureConfig

# Colonnes de forme produites côté équipe (avant préfixage home_/away_)
_FEATURE_COLUMNS = [
    "points_history",
    "winrate_history",
    "drawrate_history",
    "team_score_history",
    "opponent_score_history",
    "goal_diff_history",
    "played",
    "rest",
    "win_streak",
    "draw_streak",
]


def _to_long(res: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Passe les matchs *joués* en format long : 1 ligne par équipe et par match.

    Le `filter(finished)` est essentiel : un match non joué (score `null`) serait sinon
    compté comme une défaite et polluerait les rolling/streaks.
    """
    played = res.filter(pl.col("finished"))

    home = played.select(
        pl.col("match_id"),
        pl.col("date"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("home_score").alias("team_score"),
        pl.col("away_score").alias("opponent_score"),
        pl.lit(True).alias("is_home"),
    )
    away = played.select(
        pl.col("match_id"),
        pl.col("date"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("away_score").alias("team_score"),
        pl.col("home_score").alias("opponent_score"),
        pl.lit(False).alias("is_home"),
    )

    return (
        pl.concat([home, away])
        .with_columns(
            points=(
                pl.when(pl.col("team_score") > pl.col("opponent_score"))
                .then(cfg.victory_points)
                .when(pl.col("team_score") == pl.col("opponent_score"))
                .then(cfg.draw_points)
                .otherwise(cfg.lose_points)
            ),
            won=(pl.col("team_score") > pl.col("opponent_score")).cast(pl.Int8),
            draw=(pl.col("team_score") == pl.col("opponent_score")).cast(pl.Int8),
            goal_diff=pl.col("team_score") - pl.col("opponent_score"),
        )
        .sort("team", "date", "match_id")
    )


def add_team_form(res: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Calcule les features de forme et renvoie [match_id, home_*, away_*]."""
    long = _to_long(res, cfg)

    teams_features = (
        long
        .with_columns(
            points_history=(
                pl.col("points")
                .shift(1)
                .rolling_mean(cfg.points_history_size, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(cfg.default_points),
            winrate_history=(
                pl.col("won")
                .shift(1)
                .rolling_mean(cfg.winrate_history_size, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(cfg.default_rate),
            drawrate_history=(
                pl.col("draw")
                .shift(1)
                .rolling_mean(cfg.drawrate_history_size, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(cfg.default_rate),
            team_score_history=(
                pl.col("team_score")
                .shift(1)
                .rolling_mean(cfg.scores_history_size, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(cfg.default_scores),
            opponent_score_history=(
                pl.col("opponent_score")
                .shift(1)
                .rolling_mean(cfg.scores_history_size, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(cfg.default_scores),
            goal_diff_history=(
                pl.col("goal_diff")
                .shift(1)
                .rolling_mean(cfg.goal_diff_history_size, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(cfg.default_goal_diff),
            played=(
                pl.cum_count("match_id").over("team", order_by="date") - 1
            ),
            rest=(
                (pl.col("date") - pl.col("date").shift(1))
                .dt.total_days()
                .over("team", order_by="date")
                .clip(upper_bound=cfg.rest_cap)
                .fill_null(cfg.default_rest)
            ),
        )
        # Streaks : un groupe s'incrémente à chaque non-victoire (resp. non-nul), puis le
        # cumul des victoires (resp. nuls) DANS ce groupe donne la série en cours.
        .with_columns(
            win_grp=(1 - pl.col("won")).cum_sum().over("team", order_by="date"),
            draw_grp=(1 - pl.col("draw")).cum_sum().over("team", order_by="date"),
        )
        .with_columns(
            win_streak_incl=pl.col("won").cum_sum().over(["team", "win_grp"], order_by="date"),
            draw_streak_incl=pl.col("draw").cum_sum().over(["team", "draw_grp"], order_by="date"),
        )
        # shift(1) : streak à l'ENTRÉE du match (= valeur après le match précédent)
        .with_columns(
            win_streak=pl.col("win_streak_incl").shift(1).over("team", order_by="date").fill_null(0),
            draw_streak=pl.col("draw_streak_incl").shift(1).over("team", order_by="date").fill_null(0),
        )
        .drop("win_grp", "win_streak_incl", "draw_grp", "draw_streak_incl")
    )

    home = teams_features.filter(pl.col("is_home")).select(
        "match_id", *[pl.col(c).alias(f"home_{c}") for c in _FEATURE_COLUMNS]
    )
    away = teams_features.filter(~pl.col("is_home")).select(
        "match_id", *[pl.col(c).alias(f"away_{c}") for c in _FEATURE_COLUMNS]
    )
    return home.join(away, on="match_id")
