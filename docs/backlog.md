# Backlog — adapter le modèle à la cible « Coupe du monde »

## Pourquoi ce backlog

L'EDA des erreurs de TabPFN pur (run `5191988f…`, voir [notebooks/error_analysis.py](../notebooks/error_analysis.py))
a montré que **le backtest actuel mesure la mauvaise chose**. Le log-loss global (≈ 0,828) est dominé
par des amicaux / qualifs joués à domicile, alors que la cible d'évaluation (CDM 2026, aux États-Unis)
c'est : **catégorie *world*, terrain *neutre*, matchs souvent serrés où le nul est fréquent** — exactement
les régimes où le modèle est le plus faible :

- **world** : log-loss ≈ 0,90 (mais *n* faible sur 2025 → peu représentatif).
- **neutre** : 0,88 vs 0,80 à domicile — or la CDM est quasi 100 % neutre.
- **serré / très serré** : 1,0–1,1 (intrinsèquement dur, mais c'est le régime des poules).
- **nul** : jamais prédit en argmax — **bien que la calibration globale des probas soit correcte**.

Constat transverse : la force du modèle vient de l'**avantage du terrain**, qui **ne transfère pas** à
un tournoi neutre où `home_team`/`away_team` n'est qu'un ordre nominal.

## Deux pièges méthodologiques à respecter dans toutes les stories

1. **Features sur tout, filtrage des lignes seulement.** Les features (ELO, forme) doivent être calculées
   sur **l'intégralité** des matchs (amicaux compris) ; seul le périmètre des **lignes** train/test doit
   être restreint au compétitif. ⚠️ `make_backtest_split(categories=…)` filtre **avant** le calcul des
   features ([training/backtest.py:108](../training/backtest.py#L108)) → inadapté ici (corrigé par S2).
2. **La calibration est déjà bonne.** Tout « boost » global du nul *dégraderait* le log-loss. Le levier
   nul est donc *conditionnel* et reporté (voir S6, hors lot).

## Ordre de priorité

`S0` (réorga + couture d'évaluation) → `S1` (backtest tournoi) → `S3` / `S4` / `S5` (validés contre S1).

> **S2 est absorbée par S0.** Le « split features-sur-tout / filtre des lignes » n'est pas un chantier
> séparé : c'est le cœur de la couture d'évaluation introduite en S0 (commit 2). On garde la trace de
> S2 plus bas, marquée *fusionnée*.

---

## S0 — Réorganisation + couture d'évaluation (socle) 🧱

**Objectif** : assainir les points d'entrée et introduire **une couture d'évaluation réutilisable**
(`build features sur tout → filtre des lignes → frame de probas → score`) sur laquelle S1, le notebook
et `submit.py` se branchent, au lieu de réimplémenter chacun le fenêtrage du backtest.

**Contexte** : le cœur `training/` (data → features → backtest → model → tuning) est propre et testé.
Le « pas carré » est aux **bords** : (a) deux pipelines pandas legacy ([predict.py](../predict.py),
[baseline.py](../baseline.py)) qui dupliquent tout le feature engineering ; (b) le fenêtrage train/test
réimplémenté à la main dans [submit.py](../submit.py#L168-L178) au lieu d'appeler `make_backtest_split` ;
(c) le post-traitement des probas (clip/renorm/alignement classes) éparpillé entre `ensemble._clip_renorm`,
`submit.py` et `model.evaluate`, plus le bricolage du notebook. Cette couture corrige aussi le **piège n°1**
(features sur tout, filtre des lignes seulement).

**Découpage en commits** (chacun : `uv run pytest` vert, **zéro appel API réel**) :

- **C1 — `training/proba.py` (unifier le post-traitement).** Extraire `clip_renorm(proba, eps)` (et
  l'alignement de colonnes sur `classes_`), aujourd'hui dupliqués dans `ensemble._clip_renorm`,
  [submit.py:191-192](../submit.py#L191-L192) et la renorm de `model.evaluate`. Tous les appelants l'importent.
  *Acceptation* : un seul point de vérité pour les probas ; tests existants inchangés et verts.
- **C2 — `training/evaluation.py` (la couture = ex-S2).** `build_eval_frame(cfg, res=None)` (features sur
  **tout**, une fois) ; `select_rows(wide, *, train_pred, test_pred)` (filtre **lignes**, pas features) ;
  `predict_proba_frame(clf, test, feature_columns)` → DF (métadonnées + `p_*` alignées) ; `score(frame)`
  (log-loss + accuracy). *Acceptation* : `tests/test_evaluation.py` prouve que filtrer les **lignes** ne
  change PAS les valeurs de features (ELO d'un même match identique avec/sans filtre), contrairement au
  filtre amont `categories=` (critère d'acceptation de l'ex-S2).
- **C3 — `make_backtest_split` → wrapper.** Réécrit au-dessus de la couture, **signature conservée**
  (rétro-compat des tests). Ajoute un chemin « filtre des lignes évaluées » distinct du `categories=` amont.
  *Acceptation* : `tests/test_backtest.py` vert sans modification de comportement.
- **C4 — Rebrancher les appelants.** `submit.py` (supprime le fenêtrage dupliqué), `model.run_backtest`,
  et le notebook [error_analysis.py](../notebooks/error_analysis.py) (utilise `predict_proba_frame`).
  *Acceptation* : `submit.py --no-backtest` produit le même schéma CSV ; `uvx marimo check` du notebook OK.
- **C5 — Déplacements de fichiers.** `predict.py`/`baseline.py` → `legacy/` (+ `legacy/README.md` « réfs
  historiques, non maintenues ») ; `tests/test_baseline.py` → `tests/legacy/` avec import corrigé ;
  `submit.py`/`tune_ensemble.py` → `cli/`. Mettre à jour [README.md](../README.md) (`python predict.py`
  → `python legacy/predict.py`), `mlflow-ui.sh` si besoin, et les liens de ce backlog. Nettoyer le dossier
  vide `tests/backtest/`. *Acceptation* : `uv run pytest` vert depuis la racine, aucun import cassé.

**On ne touche pas** : `features/`, `config.py`, `mlflow_io.py`, `ensemble.py` (logique métier) — seulement
consommés. **Dépendances** : aucune ; prérequis de S1.

---

## S1 — Backtest sur compétitions (fondation) 🥇

**Objectif** : disposer d'une métrique qui ressemble à la cible — le log-loss sur de **vrais matchs de
grand tournoi** — et en faire l'objectif de tuning, à la place du log-loss global.

**Contexte** : aujourd'hui on évalue sur un cutoff calendaire (tous matchs ≥ date). On veut évaluer sur
les matchs *des tournois eux-mêmes* (CDM 2014, 2018, 2022 ; option Euro / Copa pour le volume).

**Périmètre / tâches**
- Identifier les tournois cibles. ⚠️ `load_matches` **supprime** la colonne `tournament`
  ([training/data.py](../training/data.py)) → exposer le nom du tournoi (ou définir les tournois par
  fenêtre de dates + `tournament_category_label == "world"`).
- Pour chaque tournoi : cutoff = veille du tournoi, **train sur tout l'historique antérieur**, **test =
  uniquement les matchs du tournoi**. Features calculées sur le dataset complet (cf. piège 1 / S2).
- Métrique = log-loss moyen sur les matchs du tournoi, agrégée en **leave-one-tournament-out** (éviter de
  sur-tuner sur un seul tournoi).
- Brancher cette métrique dans le tuning Optuna ([training/tuning.py](../training/tuning.py)) et/ou exposer
  une fonction réutilisable par le notebook et `submit.py`.

**Critères d'acceptation**
- Une fonction/CLI produit, pour une liste de tournois, le log-loss par tournoi + la moyenne LOTO.
- Le périmètre de test ne contient **que** des matchs du tournoi visé (vérifié sur compte de matchs ≈ 64/CDM).
- Les features d'un match de tournoi reflètent bien l'état pré-tournoi (pas de fuite, cf. builders existants).

**Dépendances** : aucune (à faire en premier). **Notes** : chaque tournoi = 1 fit TabPFN (coût API → cache
les probas comme dans le notebook ; ne lancer les fits qu'avec accord explicite).

---

## S2 — Split « features sur tout / filtre des lignes » (✅ fusionnée dans S0·C2)

> **Fusionnée dans S0 (commit C2).** Conservée ci-dessous pour la traçabilité ; ne pas planifier
> séparément. Le critère d'acceptation (filtrer les lignes ne change pas les features) est porté par
> `tests/test_evaluation.py` de S0.

**Objectif** : pouvoir restreindre les lignes train/test à un sous-ensemble de catégories **sans** fausser
le calcul des features.

**Contexte** : corrige le piège n°1. Nécessaire pour S1 (test = matchs de tournoi) et S3 (train compétitif).

**Périmètre / tâches**
- Découpler dans `make_backtest_split` (ou une variante) : `build_features(res_complet, cfg)` **puis** filtre
  des lignes `train`/`test` par prédicat (catégorie, neutralité, fenêtre tournoi).
- Garder la rétro-compat : le `categories=` actuel (filtre amont) reste possible mais documenté comme
  « filtre l'historique » ; ajouter un paramètre distinct pour « filtrer seulement les lignes évaluées ».

**Critères d'acceptation**
- Un test prouve que filtrer les *lignes* ne change pas les valeurs de features (ELO d'un match identique
  avec ou sans filtre de lignes), contrairement au filtre amont.
- S1 consomme ce mécanisme.

**Dépendances** : aucune ; prérequis de S1/S3.

---

## S3 — Dataset compétitif (world / major / qualif) 

**Objectif** : entraîner/évaluer sur le foot international compétitif plutôt que sur le bruit des amicaux.

**Périmètre / tâches**
- Définir le périmètre « compétitif » (au moins `world`, `continental_major`,
  `qualification_and_nations_leagues` — labels dans [training/data.py](../training/data.py)).
- **Éval** restreinte à ce périmètre (via S2). **Train** : tester deux variantes — (a) tout l'historique,
  (b) compétitif seulement — et comparer sur S1 (TabPFN n'a pas de pondération d'échantillons simple, donc
  on arbitre par filtrage de lignes).
- Conserver tous les matchs pour les **features** (piège 1).

**Critères d'acceptation**
- Le log-loss LOTO (S1) de la meilleure variante est **≤** celui de la baseline « train sur tout, éval globale ».
- Décision tranchée et documentée : train compétitif-only vs train complet.

**Dépendances** : S1, S2.

---

## S4 — Neutralisation de l'effet home / away 

**Objectif** : supprimer le biais « avantage à domicile » sur les matchs neutres, où home/away est arbitraire.

**Périmètre / tâches**
- **Quick win (predict-time)** : pour un match neutre, prédire `A vs B` **et** `B vs A` (features miroir),
  puis moyenner les probas (avec `p_home`/`p_away` échangés). Garantit la symétrie exacte ; remonte un peu la
  masse sur le nul. À appliquer dans `submit.py` pour la CDM.
- **Train-time (augmentation)** : dupliquer les matchs neutres avec home↔away inversés (+ label miroir) à
  l'entraînement, pour que le modèle apprenne la symétrie.
- Vérifier que les features fournies n'injectent pas un avantage positionnel résiduel (les diffs et
  `elo_diff` zéroïsent déjà `home_adv` sur neutre — cf. [training/features/__init__.py](../training/features/__init__.py)).

**Critères d'acceptation**
- Sur un match neutre, `pred(A,B)` et `pred(B,A)` donnent des probas cohérentes (home/away échangés) après
  symétrisation.
- Le log-loss LOTO (S1) ne régresse pas ; idéalement s'améliore sur le sous-ensemble neutre.

**Dépendances** : S1 (pour valider). Indépendant de S3.

---

## S5 — « Sharpen » pour la soumission 

**Objectif** : exploiter le fait qu'on ne prédit que ~18 matchs en finale → on peut se payer un modèle plus
coûteux et mieux calibré/sharp, sans contrainte de quota.

**Contexte** : `n_estimators=2` est un choix *coût* pour le tuning, pas optimal pour la soumission.

**Périmètre / tâches**
- Balayer `n_estimators` (8 / 16 / 32) et `thinking_mode` (+`thinking_effort`) — overrides déjà présents
  dans [submit.py](../submit.py) — et mesurer sur S1.
- Choisir la config soumission qui minimise le log-loss LOTO.

**Critères d'acceptation**
- Comparatif log-loss LOTO (S1) pour chaque réglage, et config retenue documentée.
- Coût API maîtrisé (fits uniquement avec accord explicite).

**Dépendances** : S1.

---

## Reporté (hors lot)

### S6 — Nul conditionnel (spike diagnostique)
À n'ouvrir qu'**après** S1. D'abord *mesurer* la calibration de la classe `draw` sur les matchs de tournoi
neutres et serrés. **Seulement si** le nul y est réellement sous-prédit : envisager un correctif *conditionnel*
(feature « buts attendus / profil défensif » de préférence ; module annexe draw/not-draw en dernier recours,
sachant que l'ensembling GBM n'a déjà rien apporté et que le risque d'overfit sur peu de données est élevé).
**Pas de boost global** (la calibration d'ensemble est déjà correcte).
