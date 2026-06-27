"""Chargement et nettoyage des données de matchs.

`load_matches()` renvoie le frame `res` (1 ligne par match) utilisé comme base par toutes
les fonctions de features. Les matchs non joués (scores `null`) sont conservés ici — ce
sont les modules de features qui filtrent sur `finished` quand ils construisent
l'historique.
"""
import polars as pl

from training.config import RESULTS_PATH, TOURNAMENT_IMPORTANCE_PATH


def load_tournament_importance() -> pl.DataFrame:
    """Charge la table d'importance des tournois (catégorie -> label + rang 1..7)."""
    return (
        pl.read_csv(TOURNAMENT_IMPORTANCE_PATH)
        .with_columns(
            (
                pl.when(pl.col("category").eq("1. Mondial"))
                .then(pl.lit("world"))
                .when(pl.col("category").eq("2. Continental majeur"))
                .then(pl.lit("continental_major"))
                .when(pl.col("category").eq("3. Qualifications & Ligue des nations"))
                .then(pl.lit("qualification_and_nations_leagues"))
                .when(pl.col("category").eq("4. Regional / sous-continental"))
                .then(pl.lit("regional"))
                .when(pl.col("category").eq("5. Amical / invitation / mineur"))
                .then(pl.lit("minor"))
                .when(pl.col("category").eq("6. Non-FIFA"))
                .then(pl.lit("non_fifa"))
                .otherwise(pl.lit("really_minor"))
                .alias("tournament_category_label")
            ),
            (
                pl.when(pl.col("category").eq("1. Mondial")).then(1)
                .when(pl.col("category").eq("2. Continental majeur")).then(2)
                .when(pl.col("category").eq("3. Qualifications & Ligue des nations")).then(3)
                .when(pl.col("category").eq("4. Regional / sous-continental")).then(4)
                .when(pl.col("category").eq("5. Amical / invitation / mineur")).then(5)
                .when(pl.col("category").eq("6. Non-FIFA")).then(6)
                .otherwise(7)
                .alias("tournament_category")
            ),
        )
        .drop("category")
    )


def load_matches() -> pl.DataFrame:
    """Charge les résultats locaux, ajoute `finished`, l'importance du tournoi et `match_id`.

    Garde `neutral` intact (servira à l'ELO). Les matchs non joués restent présents.
    """
    return (
        pl.read_csv(RESULTS_PATH, null_values=["NA"])
        .with_columns(
            (pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
            .alias("finished"),
            pl.col("date").str.to_date(),
        )
        .join(load_tournament_importance(), how="left", on="tournament")
        .drop("tournament")
        .with_row_index("match_id")
    )
