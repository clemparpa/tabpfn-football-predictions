"""Point d'entrée : entraîne et évalue TabPFN sur un split de backtest.

Construit le split (cutoff + années d'historique), entraîne le modèle, évalue sur le test et
logge le run dans MLflow. Le 1er appel à `tabpfn_client` peut demander un token
d'authentification.
"""
from datetime import date

from training.model import run_backtest

DEFAULT_CUTOFF = date(2022, 1, 1)
DEFAULT_TRAIN_YEARS = 12


def main():
    result = run_backtest(DEFAULT_CUTOFF, DEFAULT_TRAIN_YEARS)
    print(
        f"cutoff={DEFAULT_CUTOFF} train_years={DEFAULT_TRAIN_YEARS} "
        f"| train={result.n_train} test={result.n_test} "
        f"| accuracy={result.accuracy:.4f} log_loss={result.log_loss:.4f}"
    )


if __name__ == "__main__":
    main()
