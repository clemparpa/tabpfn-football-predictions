"""Configuration centrale pour le feature engineering.

Tous les hyperparamètres tunables vivent dans `FeatureConfig` (dataclass frozen) afin
d'être passés en argument aux fonctions de features. Cela permet à Optuna de faire varier
les valeurs par trial sans monkey-patcher de constantes globales.
"""
from dataclasses import dataclass

# --- Constantes non tunables (chemins / sources de données) ---
RESULTS_PATH = "results.csv"
TOURNAMENT_IMPORTANCE_PATH = "data/tournois_importance.csv"


@dataclass(frozen=True)
class FeatureConfig:
    """Hyperparamètres des features de forme par équipe.

    Repris 1:1 des constantes du notebook ; ce sont les leviers de tuning Optuna.
    """

    # Tailles des fenêtres de rolling (nombre de matchs passés)
    points_history_size: int = 5
    winrate_history_size: int = 5
    drawrate_history_size: int = 5
    scores_history_size: int = 5
    goal_diff_history_size: int = 5

    # Valeurs par défaut au cold start (équipe sans historique)
    default_points: float = 1.3
    default_rate: float = 0.33
    default_scores: float = 1.0
    default_goal_diff: float = 0.0
    default_rest: int = 30

    # Plafond du repos en jours
    rest_cap: int = 90

    # Barème de points
    victory_points: int = 3
    draw_points: int = 1
    lose_points: int = 0
