"""Tests du backtest tournoi (`training.tournament`), entièrement hors-ligne.

Aucun appel API : `evaluate_tournaments` reçoit un `fit_fn` injecté qui entraîne un
`DummyClassifier` local. Les données viennent de la fixture `real_matches` (lecture disque).
"""
import math

import polars as pl
import pytest
from sklearn.dummy import DummyClassifier

from training.evaluation import feature_matrix
from training.model import FEATURE_COLUMNS
from training.tournament import (
    DEFAULT_TOURNAMENTS,
    Tournament,
    evaluate_tournaments,
    tournament_split,
)


def _fake_fit(train: pl.DataFrame, feature_columns):
    """Entraîne un faux classifieur local (priors de classe) — zéro réseau."""
    clf = DummyClassifier(strategy="prior")
    clf.fit(feature_matrix(train, feature_columns), train.get_column("outcome").to_numpy())
    return clf


def test_test_set_is_exactly_the_tournament_edition(real_matches):
    """Le périmètre de test = uniquement les matchs de l'édition visée (≈ 64 pour une CDM)."""
    split = tournament_split(Tournament("FIFA World Cup", 2018), res=real_matches)

    assert split.test.height == 64  # 64 matchs en CDM depuis 1998
    assert (split.test.get_column("tournament") == "FIFA World Cup").all()
    assert (split.test.get_column("date").dt.year() == 2018).all()
    assert split.test.get_column("outcome").is_not_null().all()  # tous joués


def test_train_excludes_the_tournament_and_anything_after(real_matches):
    """Train = strictement pré-tournoi : rien de l'édition ni de postérieur au cutoff."""
    split = tournament_split(Tournament("FIFA World Cup", 2018), res=real_matches)

    assert split.cutoff.year == 2018 and split.cutoff.month == 6  # CDM 2018 démarre mi-juin
    assert split.train.get_column("date").max() < split.cutoff
    overlap = set(split.train.get_column("match_id")) & set(split.test.get_column("match_id"))
    assert not overlap


def test_features_reflect_full_history_not_just_the_tournament(real_matches):
    """Anti-fuite/anti-piège n°1 : un match de tournoi voit tout l'historique réel d'avant.

    `home_played` (nb de matchs joués par l'équipe à domicile avant le coup d'envoi) dépasse
    largement la taille du tournoi → les features ne sont pas amputées au seul périmètre testé.
    """
    split = tournament_split(Tournament("FIFA World Cup", 2018), res=real_matches)
    assert split.test.get_column("home_played").min() > 100


def test_missing_edition_raises(real_matches):
    with pytest.raises(ValueError, match="Aucun match joué"):
        tournament_split(Tournament("FIFA World Cup", 1800), res=real_matches)


def test_evaluate_tournaments_loto_is_mean_of_per_tournament(real_matches):
    """LOTO = moyenne des log-loss/accuracy par tournoi ; métriques valides, zéro appel API."""
    pool = (Tournament("FIFA World Cup", 2014), Tournament("FIFA World Cup", 2018))
    report = evaluate_tournaments(pool, fit_fn=_fake_fit, res=real_matches)

    assert len(report.per_tournament) == 2
    for r in report.per_tournament:
        assert r["n_test"] == 64
        assert math.isfinite(r["log_loss"])
        assert 0.0 <= r["accuracy"] <= 1.0

    expected_ll = sum(r["log_loss"] for r in report.per_tournament) / 2
    expected_acc = sum(r["accuracy"] for r in report.per_tournament) / 2
    assert report.loto_log_loss == pytest.approx(expected_ll)
    assert report.loto_accuracy == pytest.approx(expected_acc)


def test_default_pool_covers_world_cup_euro_and_copa(real_matches):
    """Le pool par défaut mélange CDM + Euro + Copa, et chaque édition existe dans les données."""
    names = {t.name for t in DEFAULT_TOURNAMENTS}
    assert names == {"FIFA World Cup", "UEFA Euro", "Copa América"}
    # Chaque édition par défaut doit se résoudre sans lever (au moins un match joué).
    for t in DEFAULT_TOURNAMENTS:
        assert tournament_split(t, res=real_matches).test.height > 0
