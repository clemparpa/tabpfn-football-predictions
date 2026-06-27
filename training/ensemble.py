"""Modèle d'ensemble : TabPFN + (optionnel) Gradient Boosting.

L'idée : TabPFN est un excellent modèle généraliste, mais un Gradient Boosting entraîné sur les
*mêmes* features peut apporter un signal complémentaire (interactions non linéaires, probas
calibrées différemment). On les combine au niveau des **probabilités** pour gratter du log-loss,
la métrique du concours.

`EnsembleClassifier` respecte l'interface sklearn (`fit`/`predict`/`predict_proba`/`classes_`)
attendue par tout le pipeline (`train_classifier`/`evaluate` dans `training.model`) : il s'insère
donc partout où un `TabPFNClassifier` était utilisé, sans rien modifier en aval. Avec ses défauts
(`use_gbm=False`, `weight=1.0`), il se comporte **exactement** comme TabPFN seul (rétro-compat).

La logique de combinaison vit dans la fonction pure `combine_probas` — partagée par la classe et
par l'objectif Optuna de `tune_ensemble.py` (qui travaille sur des probas TabPFN cachées), pour
garantir des maths **identiques** entre tuning et soumission.

`tabpfn_client` reste un import paresseux (dans `build_tabpfn`) : ce module est donc testable
hors-ligne en injectant un faux classifieur (`tabpfn=`), à l'image de `classifier=` dans
`training.model`.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from training.proba import PROBA_EPS, align_proba, clip_renorm
from training.tuning import build_tabpfn


def combine_probas(
    p_tabpfn: np.ndarray,
    p_gbm: np.ndarray,
    *,
    method: str,
    weight: float,
    eps: float = PROBA_EPS,
) -> np.ndarray:
    """Combine deux matrices de probas (mêmes colonnes/ordre de classes) et renvoie un mélange valide.

    `weight` est le poids de TabPFN ; `(1 - weight)` celui du GBM.
    - `method="arith"` : moyenne arithmétique pondérée `w·p_tabpfn + (1-w)·p_gbm`.
    - `method="geom"`  : moyenne géométrique pondérée `p_tabpfn^w · p_gbm^(1-w)` (en espace log,
      souvent mieux alignée sur le log-loss). Clip à `eps` avant le `log` pour éviter `-inf`.

    Dans tous les cas, sortie cliper+renormalisée (`0 < p < 1`, somme≈1).
    """
    if method == "arith":
        mixed = weight * p_tabpfn + (1.0 - weight) * p_gbm
    elif method == "geom":
        log_mix = weight * np.log(np.clip(p_tabpfn, eps, None)) + (1.0 - weight) * np.log(
            np.clip(p_gbm, eps, None)
        )
        mixed = np.exp(log_mix)
    else:
        raise ValueError(f"method inconnue : {method!r} (attendu 'arith' ou 'geom')")
    return clip_renorm(mixed, eps)


def build_gbm(gbm_kwargs: dict, random_state: int) -> HistGradientBoostingClassifier:
    """Construit le Gradient Boosting de l'ensemble (histogrammes, scale-invariant).

    Pas de `StandardScaler` : les arbres sont insensibles à l'échelle des features. `gbm_kwargs`
    porte les hyperparams tunés (`learning_rate`, `max_iter`, `max_depth`, `l2_regularization`).
    """
    return HistGradientBoostingClassifier(random_state=random_state, **gbm_kwargs)


@dataclass
class EnsembleConfig:
    """Configuration du modèle d'ensemble.

    Les défauts (`use_gbm=False`, `weight=1.0`) reproduisent TabPFN seul : un
    `EnsembleConfig(tabpfn_kwargs=k)` est donc équivalent à `build_tabpfn(k, ...)`.
    """

    tabpfn_kwargs: dict = field(default_factory=dict)
    use_gbm: bool = False
    combine: str = "arith"  # "arith" | "geom"
    weight: float = 1.0  # poids de TabPFN dans la combinaison ; GBM = (1 - weight)
    gbm_kwargs: dict = field(default_factory=dict)


class EnsembleClassifier:
    """Ensemble TabPFN (+ GBM optionnel), interface sklearn duck-typée par le pipeline.

    `tabpfn` est injectable (faux classifieur en test) ; sinon il est construit paresseusement
    via `build_tabpfn(cfg.tabpfn_kwargs, random_state)` au moment du `fit` (seul appel réseau).
    """

    def __init__(self, cfg: EnsembleConfig, random_state: int = 42, tabpfn=None):
        self.cfg = cfg
        self.random_state = random_state
        self._tabpfn = tabpfn
        self._gbm: HistGradientBoostingClassifier | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        y = np.asarray(y)
        if self._tabpfn is None:
            self._tabpfn = build_tabpfn(self.cfg.tabpfn_kwargs, self.random_state)
        self._tabpfn.fit(X, y)
        # Ordre de classes commun : référence unique pour aligner les probas des deux membres.
        self.classes_ = np.unique(y)
        if self.cfg.use_gbm:
            self._gbm = build_gbm(self.cfg.gbm_kwargs, self.random_state)
            self._gbm.fit(X, y)
        return self

    def _aligned_proba(self, estimator, X) -> np.ndarray:
        """Probas de `estimator` réordonnées sur `self.classes_` (robuste à un ordre différent)."""
        return align_proba(estimator.predict_proba(X), estimator.classes_, self.classes_)

    def predict_proba(self, X) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("EnsembleClassifier non entraîné : appeler fit() d'abord.")
        p_tabpfn = self._aligned_proba(self._tabpfn, X)
        if not self.cfg.use_gbm:
            # TabPFN seul : on passe quand même par combine_probas (weight=1) pour un clip/renorm
            # identique au cas ensemble — sortie strictement dans (0, 1) et sommant à 1.
            return combine_probas(p_tabpfn, p_tabpfn, method="arith", weight=1.0)
        p_gbm = self._aligned_proba(self._gbm, X)
        return combine_probas(
            p_tabpfn, p_gbm, method=self.cfg.combine, weight=self.cfg.weight
        )

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


def build_model(cfg: EnsembleConfig, random_state: int = 42) -> EnsembleClassifier:
    """Construit l'`EnsembleClassifier` (pendant de `build_tabpfn` pour le pipeline)."""
    return EnsembleClassifier(cfg, random_state=random_state)
