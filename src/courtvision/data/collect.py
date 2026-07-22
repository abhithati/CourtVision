from nba_api.stats.endpoints import leaguegamelog

log = leaguegamelog.LeagueGameLog(
    season="2023-24",
    season_type_all_star="Regular Season",
)
df = log.get_data_frames()[0]


print(df.shape)        # rows, columns — expect ~2460 rows (2 per game × ~1230 games)
print(df.columns.tolist())
df.head()

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

print(games.head())
print(games["home_win"].mean())