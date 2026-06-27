# `legacy/` — références historiques (non maintenues)

Ces scripts pandas autonomes datent de la première version du projet, **avant** le package
`training/` (polars). Ils sont conservés comme références/inspiration et ne sont **pas maintenus** :
le pipeline courant vit dans `training/`, et la soumission se génère via `python -m cli.submit`.

- **`predict.py`** — script d'origine du template : feature engineering pandas (ELO, forme, H2H),
  backtest sur le mois calendaire précédent, fit TabPFN, écriture de `predictions_YYYYMMDD.csv`.
  Lancer depuis la racine du repo :

  ```bash
  python legacy/predict.py            # prédit les fixtures à venir
  python legacy/predict.py --refresh  # rafraîchit results.csv depuis la source
  ```

- **`baseline.py`** — copie figée de `predict.py` adaptée au système de cutoff, servant de
  **garde anti-fuite** : `tests/legacy/test_baseline.py` vérifie que le pipeline polars
  (`training.backtest.make_backtest_split`) sélectionne **exactement les mêmes lignes de test**
  que ce baseline pandas. C'est sa seule raison d'être encore ici.

⚠️ Ces fichiers attendent d'être exécutés depuis la **racine** du repo (`results.csv` en chemin
relatif). Ne pas les faire évoluer : toute amélioration va dans `training/`.
