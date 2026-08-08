from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
from courtvision.config import RAW_DIR




def fetch_season(season):
    log = leaguegamelog.LeagueGameLog(
    season=season,
    season_type_all_star="Regular Season",
)
    df = log.get_data_frames()[0]

    # The teams that play at home have "vs" while the teams that are away have "@"
    home = df[df["MATCHUP"].str.contains(" vs. ")]   
    away = df[df["MATCHUP"].str.contains(" @ ")]      

    games = home.merge(away, on="GAME_ID", suffixes =("_home", "_away"))

    # This is to build the target column (1 if win, 0 for other result)
    games["home_win"] = (games["WL_home"] == "W").astype(int)

    # Indexing df w/ the column names that I want
    games = games[[
        "GAME_ID", 
        "GAME_DATE_home",
        "TEAM_ABBREVIATION_home", 
        "TEAM_ABBREVIATION_away",
        "PTS_home", 
        "PTS_away", 
        "home_win",
    ]]
    games = games.rename(columns={
        "GAME_ID": "game_id",
        "GAME_DATE_home": "game_date",
        "TEAM_ABBREVIATION_home": "home_team",
        "TEAM_ABBREVIATION_away": "away_team",
        "PTS_home": "home_pts",
        "PTS_away": "away_pts",
    })

    

    return games


def collect(seasons):
    for season in seasons:
        path = RAW_DIR / f"games_{season}.csv"
        if path.exists():
            print(f"skip {season}.csv")
            continue
        games = fetch_season(season)
        games.to_csv(path, index=False)
        print(f"saved {season}: {len(games)} games")
        time.sleep(.6)


if __name__ == "__main__":
    games = fetch_season("2023-24")
    print(games.shape)               
    print(games["home_win"].mean())  
