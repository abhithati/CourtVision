from nba_api.stats.endpoints import leaguegamelog

log = leaguegamelog.LeagueGameLog(
    season="2023-24",
    season_type_all_star="Regular Season",
)
df = log.get_data_frames()[0]


print(df.shape)        # rows, columns — expect ~2460 rows (2 per game × ~1230 games)
print(df.columns.tolist())
df.head()

