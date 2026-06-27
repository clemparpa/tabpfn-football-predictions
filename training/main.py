"""Point d'entrée : charge les données et construit le frame de features.

À ce stade le pipeline s'arrête à la construction des features. L'entraînement du modèle
(et le log MLflow) seront ajoutés une fois l'ELO et le H2H en place.
"""
from training.config import FeatureConfig
from training.data import load_matches
from training.features import build_features


def main():
    res = load_matches()
    wide = build_features(res, FeatureConfig())
    print(wide)


if __name__ == "__main__":
    main()
