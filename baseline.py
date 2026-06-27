"""Baseline TabPFN (feature engineering pandas maison) évalué sur un split de backtest.

Copie figée de l'approche `predict.py`, adaptée au même système de cutoff que le pipeline
polars (`training/model.py:run_backtest`) pour prédire **exactement les mêmes lignes de test**.
Chaque run est loggé dans une expérience MLflow séparée (`tabpfn-football-baseline`) afin de
comparer accuracy / log-loss côte à côte.

Les features de test sont calculées en chronologie native (pas de masquage) : un match de test
voit les matchs de test antérieurs — comportement authentique de predict.py, à distinguer du
pipeline polars qui gèle les features au cutoff (cf. tag MLflow `features_mode`).
"""
import argparse
import os
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tabpfn_client import TabPFNClassifier

MAX_TRAIN = 10000
HOME_ADV = 65.0
DATA = "results.csv"
RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

BASELINE_EXPERIMENT = "tabpfn-football-baseline"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # même store que training/model.py et mlflow-ui.sh

DEFAULT_CUTOFF = "2022-01-01"  # aligné sur training/main.py
DEFAULT_TRAIN_YEARS = 12

FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played",
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "neutral", "importance",
]


def importance(t):
    """Map tournament name to an ELO K-factor weight; higher means bigger rating swings."""
    t = t.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(k in t for k in ["uefa euro", "copa am", "african cup", "asian cup",
                             "gold cup", "nations league", "oceania nations"]):
        return 45.0
    if "qualif" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


def load_data(refresh=False):
    """Load and lightly clean the results CSV, downloading it if missing or refresh=True."""
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def build_features(df):
    """One chronological pass: every feature uses only matches before kickoff."""
    elo = defaultdict(lambda: 1500.0)
    res = defaultdict(list)
    last_date, h2h = {}, defaultdict(list)

    def team_feats(team):
        """Return pre-match stats for a team: ELO, form averages, win rate, goal stats, streak, games played.
        Defaults represent a mid-table team with no history."""
        r = res[team]
        if not r:
            return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
        last5, last10 = r[-5:], r[-10:]
        # each entry is (points, gf, ga, won); walk back until a non-win to count winning streak
        streak = 0
        for p, *_ in reversed(r):
            if p < 1:
                break
            streak += 1
        return (elo[team],
                np.mean([p for p, *_ in last5]), np.mean([p for p, *_ in last10]),
                np.mean([w for *_, w in last10]),
                np.mean([g for _, g, _, _ in last5]), np.mean([a for _, _, a, _ in last5]),
                np.mean([g - a for _, g, a, _ in last10]), streak, len(r))

    def h2h_feats(home, away):
        """Head-to-head record between the two teams, keyed by sorted pair so order doesn't matter.
        GD is flipped for matches where home was the away side."""
        m = h2h[tuple(sorted((home, away)))]
        if not m:
            return 0, 0.5, 0.25, 0.0
        n = len(m)
        return (n,
                sum(w == home for _, _, w in m) / n,
                sum(w == "draw" for _, _, w in m) / n,
                np.mean([g if h == home else -g for h, g, _ in m]))

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = team_feats(h)
        ae, af5, af10, awr, agf, aga, agd, astk, an = team_feats(a)
        nm, h2h_wr, h2h_dr, h2h_gd = h2h_feats(h, a)
        rows.append({
            "elo_diff": he + adj - ae, "home_elo": he, "away_elo": ae,
            "form5_diff": hf5 - af5, "form10_diff": hf10 - af10,
            "home_form5": hf5, "away_form5": af5,
            "home_winrate": hwr, "away_winrate": awr,
            "home_gf5": hgf, "away_gf5": agf, "home_ga5": hga, "away_ga5": aga,
            "gd10_diff": hgd - agd, "home_streak": hstk, "away_streak": astk,
            "home_rest": min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest": min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played": hn, "away_played": an,
            "h2h_n": nm, "h2h_home_winrate": h2h_wr, "h2h_draw_rate": h2h_dr, "h2h_gd": h2h_gd,
        })

        if not np.isnan(r.home_score):
            gd = r.home_score - r.away_score
            # standard ELO expected score from home's perspective, with home-advantage baked into adj
            exp = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            # goal-difference multiplier (FIFA-style): bigger wins shift ratings more
            g = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta
            elo[a] -= delta
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append((h, gd, h if gd > 0 else (a if gd < 0 else "draw")))

    return df.join(pd.DataFrame(rows, index=df.index))


def train(pool):
    """Fit TabPFN on the feature matrix; ignore_pretraining_limits allows >1000 rows."""
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


def backtest_split(feats, cutoff, train_years, max_train=MAX_TRAIN):
    """(pool d'entraînement, jeu de test) au cutoff donné — mêmes lignes que make_backtest_split.

    test = matchs joués avec date >= cutoff (toutes catégories) ; pool = matchs joués dans
    [cutoff - train_years, cutoff), plafonné aux `max_train` plus récents (limite TabPFN).
    """
    played = feats[feats["outcome"].notna()]
    train_start = cutoff - pd.DateOffset(years=train_years)
    pool = played[(played["date"] >= train_start) & (played["date"] < cutoff)].tail(max_train)
    test = played[played["date"] >= cutoff]
    return pool, test


def main(cutoff, train_years, max_train=MAX_TRAIN, refresh=False):
    """Backtest du baseline au cutoff donné, loggé dans l'expérience MLflow dédiée."""
    df = load_data(refresh=refresh)
    feats = build_features(df)
    pool, test = backtest_split(feats, cutoff, train_years, max_train)

    clf = train(pool)
    proba = clf.predict_proba(test[FEATURES].values)
    pred = clf.classes_[proba.argmax(1)]
    accuracy = accuracy_score(test["outcome"], pred)
    loss = log_loss(test["outcome"], proba, labels=clf.classes_)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(BASELINE_EXPERIMENT)
    with mlflow.start_run():
        mlflow.set_tags(
            {"model": "baseline_predict", "features_mode": "native_chronological"}
        )
        mlflow.log_params({
            "cutoff": str(cutoff.date()),
            "train_years": train_years,
            "n_train": len(pool),
            "n_test": len(test),
            "random_state": 42,
            "max_train": str(max_train),
        })
        mlflow.log_metrics({"accuracy": accuracy, "log_loss": loss})

    print(
        f"cutoff={cutoff.date()} train_years={train_years} "
        f"| train={len(pool)} test={len(test)} "
        f"| accuracy={accuracy:.4f} log_loss={loss:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline TabPFN sur un split de backtest")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="date de test YYYY-MM-DD")
    parser.add_argument("--train-years", type=int, default=DEFAULT_TRAIN_YEARS,
                        help="profondeur d'historique d'entraînement (années)")
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN,
                        help="plafond de lignes d'entraînement (plus récentes)")
    parser.add_argument("--refresh", action="store_true", help="re-télécharger le dataset")
    args = parser.parse_args()
    main(pd.Timestamp(args.cutoff), args.train_years, args.max_train, args.refresh)
