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

    # --- ELO ---
    elo_base: float = 1500.0
    home_adv: float = 65.0
    k_base: float = 30.0
    # Multiplicateur de K par tournament_category (1..7) ; *k_base ≈ 60/45/35/30/20/15/10.
    # cat null (tournoi inconnu) -> dernier multiplicateur (really_minor).
    k_mult_by_category: tuple[float, ...] = (2.0, 1.5, 1.17, 1.0, 0.67, 0.5, 0.33)
    elo_scale: float = 400.0  # diviseur logistique de l'écart de rating
    # Multiplicateur de buts (forme FIFA, constantes tunables) ; seuils de marge structurels.
    gd_mult_small: float = 1.0       # marge ≤ 1
    gd_mult_medium: float = 1.5      # marge = 2
    gd_mult_intercept: float = 11.0  # marge ≥ 3 : (intercept + marge) / divisor
    gd_mult_divisor: float = 8.0

    # --- H2H (valeurs par défaut au cold start, aucune confrontation passée) ---
    h2h_default_winrate: float = 0.5
    h2h_default_draw_rate: float = 0.25
    h2h_default_gd: float = 0.0
