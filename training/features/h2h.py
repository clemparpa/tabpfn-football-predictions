"""Feature Head-to-Head (bilan tête-à-tête entre deux équipes).

Contrairement à l'ELO, le H2H est vectorisable : on raisonne sur une *paire canonique*
(les deux équipes triées par ordre alphabétique, indépendamment de qui reçoit), on calcule
des cumuls par paire ordonnés par date, puis on exclut la contribution propre du match
courant pour rester pré-match (anti-leakage). Le record est ainsi disponible pour tous les
matchs, y compris les fixtures futurs (qui héritent du bilan courant de la paire).

Contrat : `add_h2h(res, cfg)` renvoie
`[match_id, h2h_n, h2h_home_winrate, h2h_draw_rate, h2h_gd]` — features symétriques au
niveau match (pas de préfixe home_/away_), exprimées du point de vue de l'équipe à domicile.
"""
import polars as pl

from training.config import FeatureConfig

# colonnes identifiant la paire canonique (équipe alphabétiquement 1re / 2de)
_PAIR_KEYS = ["team_first", "team_second"]


def add_h2h(res: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Calcule le bilan H2H pré-match et renvoie [match_id, h2h_*]."""
    home_goal_diff = pl.col("home_score") - pl.col("away_score")
    finished_int = pl.col("finished").cast(pl.Int32)

    canonical = (
        res.with_columns(
            team_first=pl.min_horizontal("home_team", "away_team"),
            team_second=pl.max_horizontal("home_team", "away_team"),
        )
        .with_columns(
            home_is_first=pl.col("home_team") == pl.col("team_first"),
            # écart de buts vu par team_first (n'a de sens que si le match est joué)
            first_goal_diff=pl.when(pl.col("home_team") == pl.col("team_first"))
            .then(home_goal_diff)
            .otherwise(-home_goal_diff)
            .fill_null(0),
            finished_int=finished_int,
        )
        .with_columns(
            first_win=((pl.col("first_goal_diff") > 0) & pl.col("finished")).cast(pl.Int32),
            is_draw=((pl.col("first_goal_diff") == 0) & pl.col("finished")).cast(pl.Int32),
        )
    )

    def prior(column: str) -> pl.Expr:
        """Cumul inclusif par paire moins la contribution propre du match = bilan pré-match."""
        return pl.col(column).cum_sum().over(
            _PAIR_KEYS, order_by=["date", "match_id"]
        ) - pl.col(column)

    accumulated = canonical.with_columns(
        prior_meetings=prior("finished_int"),
        prior_first_wins=prior("first_win"),
        prior_draws=prior("is_draw"),
        prior_first_goal_diff=prior("first_goal_diff"),
    )

    # ramène victoires et écart de buts du point de vue de l'équipe à domicile courante
    home_wins = pl.when(pl.col("home_is_first")).then(pl.col("prior_first_wins")).otherwise(
        pl.col("prior_meetings") - pl.col("prior_first_wins") - pl.col("prior_draws")
    )
    home_goal_diff_sum = pl.when(pl.col("home_is_first")).then(
        pl.col("prior_first_goal_diff")
    ).otherwise(-pl.col("prior_first_goal_diff"))

    has_history = pl.col("prior_meetings") > 0
    return accumulated.select(
        "match_id",
        h2h_n=pl.col("prior_meetings"),
        h2h_home_winrate=pl.when(has_history)
        .then(home_wins / pl.col("prior_meetings"))
        .otherwise(cfg.h2h_default_winrate),
        h2h_draw_rate=pl.when(has_history)
        .then(pl.col("prior_draws") / pl.col("prior_meetings"))
        .otherwise(cfg.h2h_default_draw_rate),
        h2h_gd=pl.when(has_history)
        .then(home_goal_diff_sum / pl.col("prior_meetings"))
        .otherwise(cfg.h2h_default_gd),
    )
