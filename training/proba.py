"""Post-traitement des probabilités : source unique de vérité.

Avant ce module, le clip/renorm et l'alignement des colonnes de classes étaient réimplémentés
à plusieurs endroits (`ensemble._clip_renorm`, `submit.py`, `model.evaluate`, le notebook
d'analyse). On centralise ici deux fonctions pures, réutilisées partout :

- `clip_renorm` : rend une matrice de probas *valide* (strictement dans (0, 1), somme 1 par
  ligne) — exigence du concours (rules.md §5) et garde-fou avant un `log` (log-loss, combinaison
  géométrique).
- `align_proba` : réordonne les colonnes d'une matrice de probas pour les faire correspondre à un
  ordre de classes cible (robuste quand deux estimateurs exposent `classes_` dans un ordre différent).
"""
import numpy as np

# Borne de sécurité pour les probas (rules.md §5 : strictement dans (0, 1)). Sert aussi de plancher
# avant un `log` (log-loss, combinaison géométrique) pour éviter les `-inf`.
PROBA_EPS = 1e-6


def clip_renorm(proba: np.ndarray, eps: float = PROBA_EPS) -> np.ndarray:
    """Clip dans `[eps, 1-eps]` puis renormalise chaque ligne à somme 1 (probas valides)."""
    proba = np.clip(proba, eps, 1.0 - eps)
    return proba / proba.sum(axis=1, keepdims=True)


def align_proba(
    proba: np.ndarray, from_classes, to_classes
) -> np.ndarray:
    """Réordonne les colonnes de `proba` (étiquetées `from_classes`) selon `to_classes`.

    `from_classes` décrit l'ordre des colonnes de `proba` (typiquement `estimator.classes_`) ;
    la sortie a ses colonnes dans l'ordre de `to_classes`. Toutes les classes de `to_classes`
    doivent exister dans `from_classes`.
    """
    from_list = list(from_classes)
    order = [from_list.index(c) for c in to_classes]
    return np.asarray(proba, dtype=np.float64)[:, order]
