import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # EDA des erreurs — TabPFN pur

    Analyse des erreurs de la run MLflow `5191988f39ec443cb55a07ea80c28084`
    (log-loss ≈ **0.828**, accuracy ≈ **62 %**, `n_estimators=2`, 24 features, `train_years=10`).

    Le concours est jugé au **log-loss** : on cible donc les matchs *confiants et faux*
    (forte contribution au log-loss), la **calibration**, et les **segments** qui sous-performent.

    > Les probas de cette run ne sont sauvegardées nulle part. On refitte TabPFN **une seule
    > fois** (1 appel API, via le bouton plus bas), on **cache** les probas en parquet, et toute
    > ré-exécution ultérieure relit le cache (100 % hors-ligne).
    """)
    return


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from datetime import date
    from pathlib import Path

    # Le notebook vit dans notebooks/ mais se lance depuis la racine du repo (cf. chemins
    # relatifs `notebooks/cache`, `results.csv`) : on y ajoute la racine pour importer `training`.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from training.backtest import make_backtest_split
    from training.model import DEFAULT_MAX_TRAIN, _feature_matrix
    from training.mlflow_io import load_run_params, reconstruct_from_params
    from training.tuning import build_tabpfn

    RUN_ID = "5191988f39ec443cb55a07ea80c28084"
    CUTOFF = date(2025, 1, 1)
    CLASSES = ["away_win", "draw", "home_win"]  # ordre canonique (np.unique des labels)
    return (
        CLASSES,
        CUTOFF,
        DEFAULT_MAX_TRAIN,
        Path,
        RUN_ID,
        alt,
        build_tabpfn,
        load_run_params,
        make_backtest_split,
        mo,
        np,
        pl,
        reconstruct_from_params,
    )


@app.cell(hide_code=True)
def _(RUN_ID, load_run_params, mo, reconstruct_from_params):
    params_run, _run_id, logged_log_loss = load_run_params(RUN_ID, None)
    cfg, feature_columns, tabpfn_kwargs, train_years = reconstruct_from_params(params_run)

    config_view = mo.md(
        f"""
        ## Config rechargée depuis MLflow

        - **log-loss loggé** : `{logged_log_loss:.4f}`
        - **train_years** : `{train_years}`
        - **tabpfn_kwargs** : `{tabpfn_kwargs}`
        - **features** ({len(feature_columns)}) : `{feature_columns}`
        - **history_size** (points) : `{cfg.points_history_size}` · home_adv `{cfg.home_adv:.1f}` ·
          k_base `{cfg.k_base:.1f}` · elo_scale `{cfg.elo_scale:.1f}`
        """
    )
    config_view
    return cfg, feature_columns, logged_log_loss, tabpfn_kwargs, train_years


@app.cell(hide_code=True)
def _(
    CUTOFF,
    DEFAULT_MAX_TRAIN,
    cfg,
    feature_columns,
    make_backtest_split,
    mo,
    train_years,
):
    from training.model import _feature_matrix

    split = make_backtest_split(CUTOFF, train_years, None, cfg)
    train_df = split.train.sort("date").tail(DEFAULT_MAX_TRAIN)
    test_df = split.test


    Xtr = _feature_matrix(train_df, feature_columns)
    ytr = train_df.get_column("outcome").to_numpy()
    Xte = _feature_matrix(test_df, feature_columns)

    split_view = mo.md(
        f"## Backtest au cutoff `{CUTOFF}`\n\n"
        f"- **train** : {train_df.height} matchs (`{split.train_start}` → `{CUTOFF}`)\n"
        f"- **test** : {test_df.height} matchs (≥ `{CUTOFF}`)"
    )
    split_view
    return Xte, Xtr, test_df, ytr


@app.cell(hide_code=True)
def _(CUTOFF, Path, RUN_ID, mo):
    cache_path = Path("notebooks/cache") / f"preds_{RUN_ID}_{CUTOFF}.parquet"
    fit_button = mo.ui.run_button(label="Fitter TabPFN (1 appel API)")

    mo.md(
        f"**Cache** : `{cache_path}` — "
        + ("présent ✅ (lecture directe)" if cache_path.exists() else "absent ⚠️ (clique pour le créer)")
    )
    return cache_path, fit_button


@app.cell(hide_code=True)
def _(fit_button):
    fit_button
    return


@app.cell(hide_code=True)
def _(
    CLASSES,
    Xte,
    Xtr,
    build_tabpfn,
    cache_path,
    fit_button,
    mo,
    np,
    pl,
    tabpfn_kwargs,
    test_df,
    ytr,
):
    mo.stop(
        not cache_path.exists() and not fit_button.value,
        mo.md("⏸️ Aucun cache trouvé — clique sur **Fitter TabPFN (1 appel API)** pour lancer le fit."),
    )

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _clf = build_tabpfn(tabpfn_kwargs, 42)
        _clf.fit(Xtr, ytr)
        _proba = np.asarray(_clf.predict_proba(Xte), dtype=float)
        _order = [list(_clf.classes_).index(c) for c in CLASSES]
        _proba = _proba[:, _order]
        (
            test_df.select(
                "match_id", "date", "home_team", "away_team", "country", "neutral",
                "tournament_category_label", "outcome",
                "elo_diff", "points_history_diff", "h2h_n", "home_played", "away_played",
            )
            .with_columns(
                p_away_win=pl.Series(_proba[:, 0]),
                p_draw=pl.Series(_proba[:, 1]),
                p_home_win=pl.Series(_proba[:, 2]),
            )
            .write_parquet(cache_path)
        )

    preds = pl.read_parquet(cache_path)
    preds
    return (preds,)


@app.cell(hide_code=True)
def _(pl, preds):
    # Frame d'analyse : prédiction (argmax), proba de la vraie classe, log-loss par match, buckets.
    analysis = (
        preds.with_columns(
            pred=pl.concat_list("p_away_win", "p_draw", "p_home_win")
            .list.arg_max()
            .cast(pl.Int64)
            .replace_strict({0: "away_win", 1: "draw", 2: "home_win"}, return_dtype=pl.Utf8),
            p_true=pl.when(pl.col("outcome") == "home_win").then(pl.col("p_home_win"))
            .when(pl.col("outcome") == "draw").then(pl.col("p_draw"))
            .otherwise(pl.col("p_away_win")),
            confidence=pl.max_horizontal("p_away_win", "p_draw", "p_home_win"),
            abs_elo_diff=pl.col("elo_diff").abs(),
        )
        .with_columns(
            correct=pl.col("pred") == pl.col("outcome"),
            match_logloss=-pl.col("p_true").clip(1e-15, 1.0).log(),
            year=pl.col("date").dt.year(),
            month=pl.col("date").dt.strftime("%Y-%m"),
        )
        .with_columns(
            conf_bucket=pl.col("confidence").cut(
                [0.4, 0.5, 0.6, 0.7, 0.8],
                labels=["<0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", ">0.8"],
            ),
            closeness=pl.col("abs_elo_diff").qcut(
                4,
                labels=["très serré", "serré", "déséquilibré", "très déséquilibré"],
                allow_duplicates=True,
            ),
            cold_start=pl.when(
                (pl.col("h2h_n") == 0)
                | (pl.min_horizontal("home_played", "away_played") < 5)
            )
            .then(pl.lit("cold-start"))
            .otherwise(pl.lit("établi")),
        )
    )
    analysis
    return (analysis,)


@app.cell(hide_code=True)
def _(analysis, logged_log_loss, mo, pl):
    g_n = analysis.height
    g_acc = analysis["correct"].mean()
    g_ll = analysis["match_logloss"].mean()
    # Baseline « prior des classes » : prédire la distribution marginale ⇒ log-loss = entropie.
    g_freqs = analysis.group_by("outcome").len().with_columns(p=pl.col("len") / g_n)
    g_entropy = -(g_freqs["p"] * g_freqs["p"].log()).sum()
    g_draw_rate = (analysis["outcome"] == "draw").mean()
    g_draw_pred = (analysis["pred"] == "draw").mean()

    metrics_view = mo.md(
        f"""
        ## Métriques globales ({g_n} matchs)

        | métrique | valeur |
        |---|---|
        | log-loss (recalculé) | **{g_ll:.4f}** |
        | log-loss (loggé MLflow) | {logged_log_loss:.4f} |
        | accuracy | **{g_acc:.1%}** |
        | baseline prior (entropie) | {g_entropy:.4f} |
        | % nuls réels | {g_draw_rate:.1%} |
        | % nuls prédits | {g_draw_pred:.1%} |

        *Léger écart possible vs le log-loss loggé si le tuning a moyenné plusieurs cutoffs.
        Si « % nuls prédits » ≪ « % nuls réels », le modèle sous-prédit structurellement les nuls.*
        """
    )
    metrics_view
    return


@app.cell(hide_code=True)
def _(analysis, mo, pl):
    mo.md("## Top 30 des plus grosses erreurs (log-loss par match)")
    top_errors = (
        analysis.sort("match_logloss", descending=True)
        .head(30)
        .select(
            "date", "home_team", "away_team", "tournament_category_label",
            "outcome", "pred",
            pl.col("p_home_win").round(3),
            pl.col("p_draw").round(3),
            pl.col("p_away_win").round(3),
            pl.col("match_logloss").round(3),
        )
    )
    mo.ui.table(top_errors, selection=None)
    return


@app.cell
def _(CLASSES, alt, analysis, mo, pl):
    # Grille 3×3 complète (outcome × pred) : TabPFN ne prédit ~jamais `draw`, sa colonne serait
    # sinon absente. On force les 9 cellules (0 % pour les manquantes) pour rendre ce fait visible.
    grid = pl.DataFrame({"outcome": CLASSES}).join(
        pl.DataFrame({"pred": CLASSES}), how="cross"
    )
    confusion = (
        grid.join(
            analysis.group_by("outcome", "pred").len(),
            on=["outcome", "pred"],
            how="left",
        )
        .with_columns(pl.col("len").fill_null(0))
        .with_columns(frac=pl.col("len") / pl.col("len").sum().over("outcome"))
    )
    _base = alt.Chart(confusion.to_pandas()).encode(
        x=alt.X("pred:N", title="prédit", sort=["home_win", "draw", "away_win"]),
        y=alt.Y("outcome:N", title="réel", sort=["home_win", "draw", "away_win"]),
    )
    confusion_chart = (
        _base.mark_rect().encode(color=alt.Color("frac:Q", title="part (par ligne)"))
        + _base.mark_text(baseline="middle").encode(
            text=alt.Text("frac:Q", format=".0%"),
            color=alt.condition("datum.frac > 0.5", alt.value("white"), alt.value("black")),
        )
    ).properties(width=300, height=300, title="Matrice de confusion (normalisée par ligne)")
    mo.vstack([mo.md("## Matrice de confusion"), confusion_chart])
    return


@app.cell(hide_code=True)
def _(CLASSES, alt, analysis, mo, pl):
    # Calibration : pour chaque classe, proba prédite (binnée) vs fréquence empirique observée.
    cal_long = (
        analysis.select("outcome", "p_away_win", "p_draw", "p_home_win")
        .unpivot(
            index="outcome",
            on=["p_away_win", "p_draw", "p_home_win"],
            variable_name="prob_col",
            value_name="p",
        )
        .with_columns(cls=pl.col("prob_col").str.replace("^p_", ""))
        .with_columns(
            actual=(pl.col("cls") == pl.col("outcome")).cast(pl.Int8),
            bin_mid=((pl.col("p") * 10).floor().clip(0, 9) / 10 + 0.05),
        )
    )
    cal_binned = (
        cal_long.group_by("cls", "bin_mid")
        .agg(
            p_pred=pl.col("p").mean(),
            freq_obs=pl.col("actual").mean(),
            n=pl.len(),
        )
        .sort("cls", "bin_mid")
    )
    _diag = alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}).to_pandas()).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q", y="y:Q")
    _enc = dict(
        x=alt.X("p_pred:Q", title="proba prédite (moyenne du bin)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("freq_obs:Q", title="fréquence observée", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("cls:N", title="classe", sort=CLASSES),
    )
    _line = alt.Chart(cal_binned.to_pandas()).mark_line(strokeWidth=2).encode(**_enc)
    _points = alt.Chart(cal_binned.to_pandas()).mark_circle(size=130, opacity=0.9).encode(
        **_enc,
        tooltip=[
            "cls",
            alt.Tooltip("p_pred:Q", format=".3f"),
            alt.Tooltip("freq_obs:Q", format=".3f"),
            "n",
        ],
    )
    calibration_chart = (_diag + _line + _points).properties(
        width=420, height=420, title="Calibration par classe (diagonale = parfait)"
    )
    mo.vstack([mo.md("## Calibration"), calibration_chart])
    return


@app.cell(hide_code=True)
def _(alt, analysis, mo, pl):
    def seg_stats(col):
        return (
            analysis.group_by(col)
            .agg(
                n=pl.len(),
                log_loss=pl.col("match_logloss").mean(),
                accuracy=pl.col("correct").mean(),
            )
            .sort("log_loss", descending=True)
            .with_columns(pl.col(col).cast(pl.Utf8))
        )

    def seg_chart(col, title):
        data = seg_stats(col).to_pandas()
        bars = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("log_loss:Q", title="log-loss moyen"),
                y=alt.Y(f"{col}:N", title=title, sort="-x"),
                color=alt.Color("accuracy:Q", scale=alt.Scale(scheme="viridis"), title="accuracy"),
                tooltip=[col, "n", alt.Tooltip("log_loss:Q", format=".3f"), alt.Tooltip("accuracy:Q", format=".1%")],
            )
            .properties(width=440, height=24 * max(len(data), 2), title=title)
        )
        return bars

    segments_view = mo.vstack(
        [
            mo.md("## Log-loss par segment\n\n*Barre = log-loss moyen (plus bas = mieux) ; couleur = accuracy.*"),
            mo.ui.tabs(
                {
                    "Catégorie tournoi": seg_chart("tournament_category_label", "catégorie de tournoi"),
                    "Terrain neutre": seg_chart("neutral", "terrain neutre"),
                    "Écart ELO (closeness)": seg_chart("closeness", "écart ELO"),
                    "Confiance": seg_chart("conf_bucket", "bucket de confiance"),
                    "Cold-start": seg_chart("cold_start", "historique équipe/H2H"),
                    "Année": seg_chart("year", "année"),
                }
            ),
        ]
    )
    segments_view
    return


@app.cell(hide_code=True)
def _(alt, analysis, mo):
    hist_chart = (
        alt.Chart(analysis.select("match_logloss", "correct").to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("match_logloss:Q", bin=alt.Bin(maxbins=40), title="log-loss du match"),
            y=alt.Y("count():Q", title="nb de matchs"),
            color=alt.Color("correct:N", title="bien classé"),
            tooltip=["count()"],
        )
        .properties(width=560, height=300, title="Distribution du log-loss par match (queue = erreurs coûteuses)")
    )
    mo.vstack([mo.md("## Distribution du log-loss par match"), hist_chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pistes de lecture

    À confronter aux graphiques ci-dessus :

    1. **Nuls** — comparer « % nuls prédits » vs « % nuls réels » et la ligne `draw` de la
       calibration : TabPFN tend à *sous-estimer* les nuls (classe la moins probable en argmax).
    2. **Sur-confiance** — sur la calibration, si la courbe passe *sous* la diagonale aux probas
       élevées (≥ 0.7), le modèle est trop confiant → forte pénalité log-loss sur les upsets.
    3. **Segments coûteux** — repérer les catégories de tournoi / buckets d'écart ELO au
       log-loss le plus élevé (souvent matchs serrés `très serré` et compétitions mineures).
    4. **Cold-start** — vérifier si `cold-start` (peu d'historique ou `h2h_n=0`) dégrade le
       log-loss : signal pour de meilleurs défauts au démarrage.
    5. **Top 30** — inspecter les upsets confiants : data leakage ? features trompeuses
       (forme/ELO) sur ces matchs précis ?
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
