import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    base_url = "https://raw.githubusercontent.com/martj42/international_results/master"
    international_results_goalscorers_path = f"{base_url}/goalscorers.csv"
    international_results_results_path = f"{base_url}/results.csv"
    tournois_importance_path = "data/tournois_importance.csv"

    goalscorers = pl.read_csv(international_results_goalscorers_path, null_values=["NA"])
    results = pl.read_csv(international_results_results_path, null_values=["NA"])
    return goalscorers, results, tournois_importance_path


@app.cell
def _(pl, tournois_importance_path):
    def load_tournament_importance():        
        return pl.read_csv(tournois_importance_path).with_columns(
            (
                pl
                .when(pl.col("category").eq("1. Mondial"))
                .then(pl.lit("world"))
                .when(pl.col("category").eq("2. Continental majeur"))
                .then(pl.lit("continental_major"))
                .when(pl.col("category").eq("3. Qualifications & Ligue des nations"))
                .then(pl.lit("qualification_and_nations_leagues"))
                .when(pl.col("category").eq("4. Regional / sous-continental"))
                .then(pl.lit("regional"))
                .when(pl.col("category").eq("5. Amical / invitation / mineur"))
                .then(pl.lit("minor"))
                .when(pl.col("category").eq("6. Non-FIFA"))
                .then(pl.lit("non_fifa"))
                .otherwise(pl.lit("really_minor")).alias("tournament_category_label")
            ),
            (
                pl
                .when(pl.col("category").eq("1. Mondial")).then(1)
                .when(pl.col("category").eq("2. Continental majeur")).then(2)
                .when(pl.col("category").eq("3. Qualifications & Ligue des nations")).then(3)
                .when(pl.col("category").eq("4. Regional / sous-continental")).then(4)
                .when(pl.col("category").eq("5. Amical / invitation / mineur")).then(5)
                .when(pl.col("category").eq("6. Non-FIFA")).then(6)
                .otherwise(7).alias("tournament_category")    
            )   
        ).drop("category")

    load_tournament_importance()
    return (load_tournament_importance,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(load_tournament_importance, pl, results):
    from datetime import date

    res = (
        results
        .with_columns(
            (pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()).alias("finished"),
            pl.col("date").str.to_date()
        )
        .join(
            load_tournament_importance(),
            how="left",
            on="tournament",
        )
        .drop("tournament")
        .with_row_index("match_id")    
    )

    home = (
        res.select(
            pl.col("match_id"), 
            pl.col("date"), 
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.col("home_score").alias("team_score"),
            pl.col("away_score").alias("opponent_score"),
            pl.lit(True).alias("is_home")
        )
    )

    away = (
        res.select(
            pl.col("match_id"), 
            pl.col("date"), 
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.col("away_score").alias("team_score"),
            pl.col("home_score").alias("opponent_score"),
            pl.lit(False).alias("is_home")
        )
    )


    VICTORY_ELO_POINTS_SCORE = 3
    LOSE_ELO_POINTS_SCORE = 0
    DRAW_ELO_POINTS_SCORE = 1

    long = (
        pl.concat([home, away])
        .with_columns(
            points=(
                pl
                .when(pl.col("team_score") > pl.col("opponent_score")).then(VICTORY_ELO_POINTS_SCORE)
                .when(pl.col("team_score") == pl.col("opponent_score")).then(DRAW_ELO_POINTS_SCORE)
                .otherwise(LOSE_ELO_POINTS_SCORE)
            ),
            won=(pl.col("team_score") > pl.col("opponent_score")).cast(pl.Int8),
            draw=(pl.col("team_score")==pl.col("opponent_score")).cast(pl.Int8),
            goal_diff=pl.col("team_score") - pl.col("opponent_score")
        )
        .sort("team", "date", "match_id")            
    )


    ELO_POINTS_HISTORY_SIZE = 5
    ELO_RATES_HISTORY_SIZE = 5
    ELO_SCORES_HISTORY_SIZE = 5
    ELO_GOAL_DIFF_HISTORY_SIZE = 5

    DEFAULT_ELO_POINTS_HISTORY = 1.3
    DEFAULT_ELO_RATE_HISTORY = 0.33
    DEFAULT_ELO_SCORES_HISTORY = 1
    DEFAULT_ELO_GOAL_DIFF_HISTORY = 0

    teams_features = (
        long
        .with_columns(
            points_history=(
                pl.col("points")
                .shift(1)
                .rolling_mean(ELO_POINTS_HISTORY_SIZE, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(DEFAULT_ELO_POINTS_HISTORY),
            winrate_history=(
                pl.col("won")
                .shift(1)
                .rolling_mean(ELO_RATES_HISTORY_SIZE, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(DEFAULT_ELO_RATE_HISTORY),     
            drawrate_history=(
                pl.col("draw")
                .shift(1)
                .rolling_mean(ELO_RATES_HISTORY_SIZE, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(DEFAULT_ELO_RATE_HISTORY),
            team_score_history=(
                pl.col("team_score")
                .shift(1)
                .rolling_mean(ELO_SCORES_HISTORY_SIZE, min_samples=1)
                .over("team", order_by="date")
            ).fill_null(DEFAULT_ELO_SCORES_HISTORY),
            opponent_score_history=(
                pl.col("opponent_score")
                .shift(1)
                .rolling_mean(ELO_SCORES_HISTORY_SIZE, min_samples=1)
                .over("team", order_by="date")          
            ).fill_null(DEFAULT_ELO_SCORES_HISTORY),
            goal_diff_history=(
                pl.col("goal_diff")
                .shift(1)
                .rolling_mean(ELO_GOAL_DIFF_HISTORY_SIZE, min_samples=1)
                .over("team", order_by="date")            
            ).fill_null(DEFAULT_ELO_GOAL_DIFF_HISTORY),
            played=(
                pl.cum_count("match_id").over("team") - 1            
            ),
            rest=(
                (pl.col("date") - pl.col("date").shift(1))
                .dt.total_days()
                .over("team", order_by="date").clip(upper_bound=90).fill_null(30)
            )
        )
        .with_columns(
            win_grp=(1 - pl.col("won")).cum_sum().over("team", order_by="date"),
            draw_grp=(1 - pl.col("draw")).cum_sum().over("team", order_by="date")
        )
        .with_columns(
            win_streak_incl=pl.col("won").cum_sum().over(["team", "win_grp"], order_by="date"),
            draw_streak_incl=pl.col("draw").cum_sum().over(["team", "draw_grp"], order_by="date")        
        )
        .with_columns(
            win_streak=pl.col("win_streak_incl").shift(1).over("team", order_by="date").fill_null(0),
            draw_streak=pl.col("draw_streak_incl").shift(1).over("team", order_by="date").fill_null(0)        
        )
        .drop("win_grp", "win_streak_incl", "draw_grp", "draw_streak_incl")
    )

    columns = [
        "points_history", "winrate_history", "drawrate_history", "team_score_history","opponent_score_history", 
        "goal_diff_history", "played", "rest", "win_streak", "draw_streak"
    ]

    wide = (
        res
        .join(
            teams_features
            .filter(pl.col("is_home"))
            .select("match_id", *[pl.col(c).alias(f"home_{c}") for c in columns]),
            on="match_id"
        )
        .join(
            teams_features
            .filter(~pl.col("is_home"))
            .select("match_id", *[pl.col(c).alias(f"away_{c}") for c in columns]),
            on="match_id"
        )
        .with_columns(
            points_history_diff=pl.col("home_points_history") - pl.col("away_points_history"),
            goal_diff_history_diff=pl.col("home_goal_diff_history") - pl.col("away_goal_diff_history")
        )
    )

    wide
    #    .filter(
    #        pl.col("date") > date(year=2000, month=1, day=1)
    #    )
    #).select("tournament").unique()
    return


@app.cell
def _():
    return


@app.cell
def _(former_names):
    former_names
    return


@app.cell
def _(goalscorers):
    goalscorers
    return


@app.cell
def _(results):
    results
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
