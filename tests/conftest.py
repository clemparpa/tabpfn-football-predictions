"""Fixtures partagées pour les tests du pipeline de features."""
from datetime import date

import polars as pl
import pytest

from training.config import FeatureConfig
from training.data import load_matches


@pytest.fixture(scope="session")
def real_matches() -> pl.DataFrame:
    """Le frame `res` chargé une seule fois depuis results.csv (vraies données)."""
    return load_matches()


@pytest.fixture(scope="session")
def default_cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture
def make_res():
    """Factory : construit un `res` synthétique à la forme produite par load_matches().

    `matches` est une liste de tuples (date, home_team, away_team, home_score, away_score).
    Un score à None marque un match non joué (finished == False). `match_id` est ajouté
    dans l'ordre de la liste.
    """

    def _make(matches: list[tuple]) -> pl.DataFrame:
        return (
            pl.DataFrame(
                matches,
                schema=[
                    ("date", pl.Date),
                    ("home_team", pl.Utf8),
                    ("away_team", pl.Utf8),
                    ("home_score", pl.Int64),
                    ("away_score", pl.Int64),
                ],
                orient="row",
            )
            .with_columns(
                (pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
                .alias("finished")
            )
            .with_row_index("match_id")
        )

    return _make


@pytest.fixture
def d():
    """Petit helper pour écrire des dates lisiblement dans les fixtures synthétiques."""
    return date
