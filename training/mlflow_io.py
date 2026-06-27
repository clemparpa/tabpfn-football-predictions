"""Lecture et reconstruction d'hyperparamètres depuis MLflow.

Partagé par `submit.py` (soumission) et `tune_ensemble.py` (tuning de l'ensemble), qui ont tous
deux besoin de recharger une run loggée par le tuning et d'en reconstituer les objets Python
(`FeatureConfig`, colonnes, kwargs TabPFN, `EnsembleConfig`).

Deux schémas de params coexistent dans le store :
- **tuning** (`training.tuning`) : préfixes `cfg.*` et `tabpfn.*`, `feature_columns`, `train_years` ;
- **backtest** (`training.model`) : champs `FeatureConfig` à plat, sans bloc `tabpfn.*`.
La reconstruction gère les deux. Les runs d'ensemble (`tune_ensemble.py`) ajoutent un bloc
`ensemble.*` et recopient les params figés ci-dessus pour rester **autonomes**.
"""
import ast
from dataclasses import fields

from training.config import FeatureConfig
from training.ensemble import EnsembleConfig
from training.model import MLFLOW_TRACKING_URI


def _parse(value: str):
    """Convertit une valeur de param MLflow (toujours str) en Python natif.

    `literal_eval` couvre ints/floats/bools/tuples/listes ; on retombe sur la chaîne brute
    pour les valeurs non littérales (ex. `thinking_effort="medium"`).
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def reconstruct_from_params(params: dict) -> tuple[FeatureConfig, tuple[str, ...], dict, int]:
    """Reconstruit (cfg, feature_columns, tabpfn_kwargs, train_years) depuis les params MLflow.

    Gère les deux schémas loggés : tuning (`cfg.*`/`tabpfn.*`) et backtest (champs `cfg` à plat,
    sans `tabpfn.*`).
    """
    # FeatureConfig : pour chaque champ, on tente `cfg.<nom>` puis `<nom>` (schéma à plat).
    cfg_kwargs = {}
    for f in fields(FeatureConfig):
        raw = params.get(f"cfg.{f.name}", params.get(f.name))
        if raw is not None:
            cfg_kwargs[f.name] = _parse(raw)
    cfg = FeatureConfig(**cfg_kwargs)

    if "feature_columns" not in params:
        raise ValueError(
            "Param `feature_columns` absent de la run MLflow : impossible de reconstruire les colonnes."
        )
    feature_columns = tuple(_parse(params["feature_columns"]))

    tabpfn_kwargs = {
        k[len("tabpfn."):]: _parse(v) for k, v in params.items() if k.startswith("tabpfn.")
    }

    if "train_years" not in params:
        raise ValueError("Param `train_years` absent de la run MLflow.")
    train_years = int(_parse(params["train_years"]))

    return cfg, feature_columns, tabpfn_kwargs, train_years


def reconstruct_ensemble_config(params: dict, tabpfn_kwargs: dict) -> EnsembleConfig:
    """Reconstruit l'`EnsembleConfig` depuis les params `ensemble.*`/`gbm.*` (+ kwargs TabPFN figés).

    Si aucun param `ensemble.*` n'est présent (run « source » antérieure à l'ensemble), renvoie
    une config TabPFN pur (`use_gbm=False`) : la run reste consommable telle quelle.
    """
    cfg = EnsembleConfig(tabpfn_kwargs=dict(tabpfn_kwargs))
    ens = {
        k[len("ensemble."):]: _parse(v) for k, v in params.items() if k.startswith("ensemble.")
    }
    if "use_gbm" in ens:
        cfg.use_gbm = bool(ens["use_gbm"])
    if "combine" in ens:
        cfg.combine = ens["combine"]
    if "weight" in ens:
        cfg.weight = float(ens["weight"])
    # Hyperparams GBM : bloc `gbm.*` (miroir de `tabpfn.*`).
    cfg.gbm_kwargs = {
        k[len("gbm."):]: _parse(v) for k, v in params.items() if k.startswith("gbm.")
    }
    return cfg


def load_run_params(
    run_id: str | None, experiment: str | None
) -> tuple[dict, str, float | None]:
    """Charge les params d'une run MLflow (par id, ou meilleure run d'un experiment par log-loss).

    Renvoie `(params, run_id, log_loss)`.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    if run_id:
        run = client.get_run(run_id)
    else:
        assert experiment is not None  # run_id falsy => experiment fourni (garanti par l'appelant)
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            raise ValueError(f"Experiment MLflow introuvable : {experiment!r}")
        runs = client.search_runs(
            [exp.experiment_id], order_by=["metrics.log_loss ASC"], max_results=1
        )
        if not runs:
            raise ValueError(f"Aucune run dans l'experiment {experiment!r}")
        run = runs[0]
    return run.data.params, run.info.run_id, run.data.metrics.get("log_loss")
