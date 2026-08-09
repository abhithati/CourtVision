from pathlib import Path

import pandas as pd

from courtvision.config import RAW_DIR


def load_games():
    """Load and concatenate every season CSV into one game-level table."""
    files = sorted(RAW_DIR.glob("games_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


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
    return team_games.sort_values(["team", "game_date"]).reset_index(drop=True)

if __name__ == "__main__":
    games = load_games()
    team_games = to_team_games(games)
    print(team_games.shape)          # expect ~(23938, 8) — 2× your 11,969 games
    print(team_games.head(10).to_string())
