import pandas as pd

from courtvision.config import PROCESSED_DIR, RAW_DIR

# Pre-tipoff features computed per team, mirrored into home_/away_ columns.
FEATURE_COLS = [
    "win_pct_5",
    "win_pct_10",
    "pts_5",
    "pts_allowed_5",
    "net_rating_5",
    "rest_days",
]


def load_games():
    """Load and concatenate every season CSV into one game-level table."""
    files = sorted(RAW_DIR.glob("games_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def season_start_year(dates):
    """Starting year of the NBA season a date falls in (2016 for "2016-17").

    Seasons run Oct-June, so months Sep-Dec belong to that calendar year's
    season and Jan-Jun belong to the previous year's.
    """
    return dates.dt.year.where(dates.dt.month >= 9, dates.dt.year - 1)


def to_team_games(games):
    """One row per game → two rows per game (each team's perspective)."""
    home = pd.DataFrame({
        "game_id": games["game_id"],
        "game_date": games["game_date"],
        "team": games["home_team"],
        "opponent": games["away_team"],
        "is_home": 1,
        "pts": games["home_pts"],
        "pts_allowed": games["away_pts"],
        "won": games["home_win"],
    })
    away = pd.DataFrame({
        "game_id": games["game_id"],
        "game_date": games["game_date"],
        "team": games["away_team"],
        "opponent": games["home_team"],
        "is_home": 0,
        "pts": games["away_pts"],
        "pts_allowed": games["home_pts"],
        "won": 1 - games["home_win"],
    })
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"])
    team_games["season"] = season_start_year(team_games["game_date"])
    return team_games.sort_values(["team", "season", "game_date"]).reset_index(drop=True)


def add_rolling_features(team_games):
    """Add rolling form features using prior games in the same season only.

    Grouping by season (not just team) resets each window every October:
    form doesn't carry across an offseason of trades and draft picks, and it
    keeps rest_days from reporting the ~190-day gap between seasons.
    """
    grp = team_games.groupby(["team", "season"])

    # prev 5 and 10 game win percentage
    team_games["win_pct_5"] = grp["won"].transform(lambda s: s.shift(1).rolling(5).mean())
    team_games["win_pct_10"] = grp["won"].transform(lambda s: s.shift(1).rolling(10).mean())

    # Scoring ability over the previous 5 games
    team_games["pts_5"] = grp["pts"].transform(lambda s: s.shift(1).rolling(5).mean())
    team_games["pts_allowed_5"] = grp["pts_allowed"].transform(lambda s: s.shift(1).rolling(5).mean())

    # Point differential — usually the strongest single form signal
    team_games["net_rating_5"] = team_games["pts_5"] - team_games["pts_allowed_5"]

    # Days since this team's previous game (NaN for their first game of a season).
    # Clipped at 7: beyond a week the exact number stops mattering, and it keeps
    # the 2020 COVID-bubble restart (~144-day mid-season gap) from skewing the scale.
    team_games["rest_days"] = grp["game_date"].diff().dt.days.clip(upper=7)

    return team_games


def to_game_level(team_games):
    """Collapse team rows back to one row per game: home features vs away features."""
    home_cols = ["game_id", "game_date", "season", "team", "won"] + FEATURE_COLS
    away_cols = ["game_id", "team"] + FEATURE_COLS

    home = team_games.loc[team_games["is_home"] == 1, home_cols]
    away = team_games.loc[team_games["is_home"] == 0, away_cols]

    games = home.merge(away, on="game_id", suffixes=("_home", "_away"))
    games = games.rename(columns={
        "team_home": "home_team",
        "team_away": "away_team",
        "won": "home_win",
    })

    return games.sort_values("game_date").reset_index(drop=True)


def build_features(save=True):
    """Full pipeline: raw CSVs → model-ready game-level feature table."""
    team_games = add_rolling_features(to_team_games(load_games()))
    model_df = to_game_level(team_games)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        model_df.to_csv(PROCESSED_DIR / "game_features.csv", index=False)

    return model_df


if __name__ == "__main__":
    df = build_features()
    print("shape:", df.shape)
    print("seasons:", sorted(df["season"].unique()))
    print("rows with any missing feature:", df.isna().any(axis=1).sum())
    print(df.head(5).to_string())
