"""Entraînement et évaluation du modèle TabPFN sur un split de backtest.

On part du `BacktestSplit` produit par `training.backtest` (frames `train`/`test` sans fuite
temporelle), on entraîne un `TabPFNClassifier` sur les features candidates, on prédit sur le
test et on mesure accuracy / log-loss. Chaque run est tracé dans MLflow afin de comparer des
configurations (et, plus tard, de brancher Optuna sur `FeatureConfig`).

Référence : `predict.py` (pandas, isolé) — labels `home_win`/`away_win`/`draw`,
`TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)`,
`accuracy_score` + `log_loss(labels=clf.classes_)`.

`tabpfn_client` est une API distante : l'entraînement envoie les données au service (token et
quotas requis, d'où le plafond `max_train`). Les tests injectent un faux classifieur pour
rester hors-ligne ; seul un appel manuel touche le réseau.
"""
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, log_loss

from training.backtest import make_backtest_split
from training.config import FeatureConfig
from training.features.team_form import _FEATURE_COLUMNS

# Colonnes données au modèle : toute la forme par équipe + ELO + H2H + diffs + contexte.
# Dérivé de `_FEATURE_COLUMNS` (source de vérité partagée) pour rester aligné sur le pipeline.
FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"{side}_{col}" for col in _FEATURE_COLUMNS for side in ("home", "away")),
    "home_elo", "away_elo",
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "points_history_diff", "goal_diff_history_diff", "elo_diff",
    "neutral",
)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # même store que mlflow-ui.sh
DEFAULT_EXPERIMENT = "tabpfn-football-backtest"
DEFAULT_MAX_TRAIN = 10_000  # limite de tabpfn_client (cf. MAX_TRAIN de predict.py)


@dataclass(frozen=True)
class BacktestResult:
    """Résultat d'un run d'entraînement + évaluation sur un split de backtest."""

    classifier: object
    accuracy: float
    log_loss: float
    n_train: int
    n_test: int
    classes: tuple[str, ...]


def _feature_matrix(df: pl.DataFrame) -> np.ndarray:
    """Extrait la matrice de features (Float64 uniforme : `neutral`/ints/floats homogènes)."""
    return df.select(pl.col(c).cast(pl.Float64) for c in FEATURE_COLUMNS).to_numpy()


def train_classifier(
    train_df: pl.DataFrame, classifier=None, random_state: int = 42
):
    """Entraîne un classifieur sur `train_df` (features + label `outcome`).

    `classifier` injectable (défaut : `TabPFNClassifier`) — permet de tester l'orchestration
    avec un faux modèle, sans appeler l'API distante.
    """
    if classifier is None:
        from tabpfn_client import TabPFNClassifier

        classifier = TabPFNClassifier(
            ignore_pretraining_limits=True, random_state=random_state
        )
    classifier.fit(_feature_matrix(train_df), train_df.get_column("outcome").to_numpy())
    return classifier


def evaluate(classifier, test_df: pl.DataFrame) -> dict:
    """Prédit sur `test_df` et renvoie accuracy / log-loss / taille du test."""
    features = _feature_matrix(test_df)
    truth = test_df.get_column("outcome").to_numpy()
    proba = classifier.predict_proba(features)
    return {
        "accuracy": accuracy_score(truth, classifier.predict(features)),
        # labels=classes_ : aligne les colonnes de proba même si une classe manque au train
        "log_loss": log_loss(truth, proba, labels=classifier.classes_),
        "n_test": test_df.height,
    }


def _log_to_mlflow(
    cfg: FeatureConfig,
    metrics: dict,
    *,
    cutoff: date,
    train_years: int,
    categories,
    n_train: int,
    random_state: int,
    max_train,
    experiment: str,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        params = {k: str(v) for k, v in asdict(cfg).items()}  # tuples -> str
        params.update(
            cutoff=str(cutoff),
            train_years=train_years,
            categories=str(categories),
            n_train=n_train,
            n_test=metrics["n_test"],
            random_state=random_state,
            max_train=str(max_train),
        )
        mlflow.log_params(params)
        mlflow.log_metrics(
            {"accuracy": metrics["accuracy"], "log_loss": metrics["log_loss"]}
        )


def run_backtest(
    cutoff: date,
    train_years: int,
    categories=None,
    cfg: FeatureConfig = FeatureConfig(),
    *,
    random_state: int = 42,
    max_train: int | None = DEFAULT_MAX_TRAIN,
    classifier=None,
    log_mlflow: bool = True,
    experiment: str = DEFAULT_EXPERIMENT,
) -> BacktestResult:
    """Construit le split, entraîne, évalue, et (option) logge le run dans MLflow.

    `max_train` plafonne l'entraînement aux lignes les plus récentes (limite de l'API TabPFN).
    `classifier` injectable et `log_mlflow=False` permettent des tests hors-ligne.
    """
    split = make_backtest_split(cutoff, train_years, categories, cfg)

    train = split.train
    if max_train is not None and train.height > max_train:
        train = train.sort("date").tail(max_train)  # garder les plus récents

    fitted = train_classifier(train, classifier, random_state)
    metrics = evaluate(fitted, split.test)

    if log_mlflow:
        _log_to_mlflow(
            cfg, metrics,
            cutoff=cutoff, train_years=train_years, categories=split.categories,
            n_train=train.height, random_state=random_state, max_train=max_train,
            experiment=experiment,
        )

    return BacktestResult(
        classifier=fitted,
        accuracy=metrics["accuracy"],
        log_loss=metrics["log_loss"],
        n_train=train.height,
        n_test=metrics["n_test"],
        classes=tuple(fitted.classes_),
    )
