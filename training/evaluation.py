"""Couture d'évaluation réutilisable : features sur tout → filtre des lignes → probas → score.

Avant ce module, le fenêtrage d'évaluation était réimplémenté à plusieurs endroits
(`make_backtest_split`, `submit.py`, le notebook). On centralise ici quatre fonctions composables :

- `build_eval_frame(cfg, res)` — calcule les features sur **l'intégralité** des matchs (une fois),
  joint le label `outcome`. C'est l'étape « features sur tout » du piège n°1 du backlog : les
  features (ELO, forme, H2H) voient tout l'historique, amicaux compris.
- `select_rows(wide, *, train_pred, test_pred)` — restreint les **lignes** train/test par prédicat
  polars. Filtrer des lignes ne change **pas** les valeurs de features déjà calculées (contrairement
  à un filtre amont type `categories=`, qui ampute l'historique avant le calcul).
- `predict_proba_frame(clf, test, feature_columns)` — prédit et renvoie une frame
  (métadonnées + `p_home_win/p_draw/p_away_win`), probas valides et alignées sur l'ordre canonique.
- `score(frame)` — log-loss + accuracy à partir d'une frame de probas.

`feature_matrix` (extraction de la matrice X) vit ici, à un niveau bas que `model`/`submit`/le
notebook réutilisent : ainsi `backtest` peut s'appuyer sur cette couture sans cycle d'import.
"""
from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, log_loss

from training.config import FeatureConfig
from training.data import load_matches
from training.features import build_features
from training.proba import align_proba, clip_renorm

# Ordre canonique des classes : `np.unique` trie alphabétiquement, donc c'est l'ordre de
# `classifier.classes_` d'un modèle entraîné sur les 3 issues. Fixe le mapping colonne ↔ classe.
CLASSES: tuple[str, ...] = ("away_win", "draw", "home_win")

# Colonnes d'identification reportées telles quelles dans la frame de probas (si présentes).
_META_COLUMNS: tuple[str, ...] = ("match_id", "date", "home_team", "away_team", "outcome")


def feature_matrix(df: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Matrice de features (Float64 uniforme) restreinte à `columns`, dans l'ordre donné."""
    return df.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()


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


def build_eval_frame(
    cfg: FeatureConfig = FeatureConfig(), res: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Calcule les features sur **tout** `res` (une fois) et joint le label `outcome`.

    `res` injectable (défaut `load_matches()`). Aucun filtre de catégorie ici : l'historique
    complet alimente les features (cf. piège n°1). Le filtrage des lignes se fait ensuite via
    `select_rows`. Les matchs non joués ont `outcome` null (left join).
    """
    if res is None:
        res = load_matches()
    labels = add_outcome(res)
    return build_features(res, cfg).join(labels, on="match_id", how="left")


def select_rows(
    wide: pl.DataFrame, *, train_pred: pl.Expr, test_pred: pl.Expr
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Restreint `wide` aux lignes train/test par prédicat polars (filtre de **lignes**).

    Ne touche pas aux features : un même match garde des valeurs identiques avec ou sans filtre.
    """
    return wide.filter(train_pred), wide.filter(test_pred)


def predict_proba_frame(
    clf, test: pl.DataFrame, feature_columns: Sequence[str]
) -> pl.DataFrame:
    """Prédit sur `test` et renvoie `[métadonnées..., p_home_win, p_draw, p_away_win]`.

    Les probas sont alignées sur l'ordre canonique `CLASSES` (robuste à l'ordre de `clf.classes_`)
    puis cliper/renormalisées (valides : `0 < p < 1`, somme 1).
    """
    proba = clf.predict_proba(feature_matrix(test, feature_columns))
    proba = clip_renorm(align_proba(proba, clf.classes_, CLASSES))
    meta = [c for c in _META_COLUMNS if c in test.columns]
    return test.select(meta).with_columns(
        p_away_win=pl.Series(proba[:, 0]),
        p_draw=pl.Series(proba[:, 1]),
        p_home_win=pl.Series(proba[:, 2]),
    )


def score(frame: pl.DataFrame) -> dict:
    """Métriques d'une frame de probas (issue de `predict_proba_frame`) : accuracy + log-loss.

    L'accuracy se lit sur l'argmax des probas (cohérent avec la prédiction de soumission) ;
    `labels=CLASSES` aligne les 3 colonnes même si une classe manque au jeu de test.
    """
    truth = frame.get_column("outcome").to_numpy()
    proba = frame.select("p_away_win", "p_draw", "p_home_win").to_numpy()
    pred = np.asarray(CLASSES)[proba.argmax(axis=1)]
    return {
        "accuracy": accuracy_score(truth, pred),
        "log_loss": log_loss(truth, proba, labels=list(CLASSES)),
        "n_test": frame.height,
    }
