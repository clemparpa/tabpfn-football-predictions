"""Features de forme par équipe (rolling sur l'historique récent).

Pour les matchs *joués*, l'anti-leakage repose sur un `shift(1)` appliqué avant chaque
`rolling_mean` : seules les rencontres antérieures au coup d'envoi entrent dans la fenêtre.

Pour les matchs *futurs* (non joués), on reporte la dernière forme connue de chaque équipe,
comme le font l'ELO et le H2H (et comme le baseline `predict.py`). On utilise alors la
variante *inclusive* (sans `shift`) — la forme telle qu'elle est juste après le dernier
match joué — récupérée par un `join_asof` sur la date. Sans report, un fixture futur n'aurait
aucune feature de forme et ne serait pas prédictible.

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


# Features de moyenne glissante : (nom produit, colonne source, taille de fenêtre, défaut cold start)
_HISTORY_SPECS = [
    ("points_history", "points", "points_history_size", "default_points"),
    ("winrate_history", "won", "winrate_history_size", "default_rate"),
    ("drawrate_history", "draw", "drawrate_history_size", "default_rate"),
    ("team_score_history", "team_score", "scores_history_size", "default_scores"),
    ("opponent_score_history", "opponent_score", "scores_history_size", "default_scores"),
    ("goal_diff_history", "goal_diff", "goal_diff_history_size", "default_goal_diff"),
]


def add_team_form(res: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Calcule les features de forme et renvoie [match_id, home_*, away_*].

    Les matchs joués reçoivent la forme pré-match (shift(1)) ; les matchs futurs héritent
    de la dernière forme connue de chaque équipe (variante inclusive, sans shift).
    """
    long = _to_long(res, cfg)

    def rolling(source: str, window: int, *, shift: bool) -> pl.Expr:
        """Moyenne glissante par équipe ; `shift=True` exclut le match courant (pré-match)."""
        column = pl.col(source).shift(1) if shift else pl.col(source)
        return column.rolling_mean(window, min_samples=1).over("team", order_by="date")

    teams_features = (
        long
        .with_columns(
            # pré-match (shift(1)) pour les matchs joués
            *[
                rolling(source, getattr(cfg, window), shift=True)
                .fill_null(getattr(cfg, default))
                .alias(name)
                for name, source, window, default in _HISTORY_SPECS
            ],
            # inclusive (sans shift) : forme juste après le dernier match joué, pour le report futur
            *[
                rolling(source, getattr(cfg, window), shift=False).alias(f"{name}_incl")
                for name, source, window, _ in _HISTORY_SPECS
            ],
            played=(pl.cum_count("match_id").over("team", order_by="date") - 1),
            played_incl=pl.cum_count("match_id").over("team", order_by="date"),
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
        .drop("win_grp", "draw_grp")
    )

    played_features = teams_features.select("match_id", "is_home", *_FEATURE_COLUMNS)
    future_features = _future_form(res, teams_features, cfg)

    all_features = pl.concat([played_features, future_features], how="vertical_relaxed")
    home = all_features.filter(pl.col("is_home")).select(
        "match_id", *[pl.col(c).alias(f"home_{c}") for c in _FEATURE_COLUMNS]
    )
    away = all_features.filter(~pl.col("is_home")).select(
        "match_id", *[pl.col(c).alias(f"away_{c}") for c in _FEATURE_COLUMNS]
    )
    return home.join(away, on="match_id")


def _future_form(
    res: pl.DataFrame, teams_features: pl.DataFrame, cfg: FeatureConfig
) -> pl.DataFrame:
    """Reporte sur chaque match futur la dernière forme connue de l'équipe.

    `latest` = état inclusive du dernier match joué de chaque équipe ; on le joint à chaque
    fixture par équipe. Une équipe sans historique retombe sur les défauts cfg. Renvoie
    [match_id, is_home, *_FEATURE_COLUMNS], homogène avec la branche « joués ».
    """
    incl_history_names = [f"{name}_incl" for name, *_ in _HISTORY_SPECS]
    latest = (
        teams_features.sort("date", "match_id")
        .unique(subset="team", keep="last", maintain_order=True)
        .select(
            "team",
            pl.col("date").alias("last_played_date"),
            *incl_history_names,
            "played_incl",
            "win_streak_incl",
            "draw_streak_incl",
        )
    )

    future = res.filter(~pl.col("finished"))
    future_long = pl.concat(
        [
            future.select(
                "match_id", "date",
                pl.col("home_team").alias("team"), pl.lit(True).alias("is_home"),
            ),
            future.select(
                "match_id", "date",
                pl.col("away_team").alias("team"), pl.lit(False).alias("is_home"),
            ),
        ]
    )

    return (
        future_long
        .join(latest, on="team", how="left")
        .with_columns(
            *[
                pl.col(f"{name}_incl").fill_null(getattr(cfg, default)).alias(name)
                for name, _, _, default in _HISTORY_SPECS
            ],
            played=pl.col("played_incl").fill_null(0),
            win_streak=pl.col("win_streak_incl").fill_null(0),
            draw_streak=pl.col("draw_streak_incl").fill_null(0),
            rest=(
                (pl.col("date") - pl.col("last_played_date"))
                .dt.total_days()
                .clip(upper_bound=cfg.rest_cap)
                .fill_null(cfg.default_rest)
            ),
        )
        .select("match_id", "is_home", *_FEATURE_COLUMNS)
    )
