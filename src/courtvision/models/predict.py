"""CLI for pre-game win probability: give two teams, get a prediction.

Usage:
    python -m courtvision.models.predict --home BOS --away LAL
    python -m courtvision.models.predict --home OKC --away UTA --home-rest 1

Form is read from each team's most recent completed games in the latest season
available, so predictions reflect current form.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from courtvision.config import MODELS_DIR
from courtvision.features.build import FEATURE_COLS, load_games, to_team_games
from courtvision.models.train import FEATURE_NAMES

MODEL_PATH = MODELS_DIR / "logreg_pregame.joblib"

# League-median rest; used when the caller doesn't specify a real gap.
DEFAULT_REST_DAYS = 2


def latest_team_games(season=None):
    """Team-level game rows for the season used to measure current form."""
    team_games = to_team_games(load_games())
    season = team_games["season"].max() if season is None else season
    return team_games[team_games["season"] == season], season


def team_form(team_games, team, rest_days):
    """Current pre-game features for one team, from their most recent games.

    Unlike training (where features must exclude the game being predicted),
    a future game's window is simply the team's last N completed games.
    """
    hist = team_games[team_games["team"] == team].sort_values("game_date")
    if hist.empty:
        teams = ", ".join(sorted(team_games["team"].unique()))
        sys.exit(f"No games found for '{team}'. Available teams:\n{teams}")

    last5, last10 = hist.tail(5), hist.tail(10)
    pts_5 = last5["pts"].mean()
    pts_allowed_5 = last5["pts_allowed"].mean()

    return {
        "win_pct_5": last5["won"].mean(),
        "win_pct_10": last10["won"].mean(),
        "pts_5": pts_5,
        "pts_allowed_5": pts_allowed_5,
        "net_rating_5": pts_5 - pts_allowed_5,
        "rest_days": rest_days,
    }, len(hist)


def predict(home, away, home_rest=DEFAULT_REST_DAYS, away_rest=DEFAULT_REST_DAYS, season=None):
    """Return (home_win_probability, context_dict) for a hypothetical matchup."""
    if not MODEL_PATH.exists():
        sys.exit(f"No trained model at {MODEL_PATH}. Run: python -m courtvision.models.train")

    model = joblib.load(MODEL_PATH)
    team_games, season_used = latest_team_games(season)

    home_feats, home_n = team_form(team_games, home, home_rest)
    away_feats, away_n = team_form(team_games, away, away_rest)

    row = {f"{col}_home": home_feats[col] for col in FEATURE_COLS}
    row.update({f"{col}_away": away_feats[col] for col in FEATURE_COLS})
    X = pd.DataFrame([row])[FEATURE_NAMES]

    prob = float(model.predict_proba(X)[0, 1])
    return prob, {
        "season": season_used,
        "home_feats": home_feats,
        "away_feats": away_feats,
        "home_games": home_n,
        "away_games": away_n,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict NBA pre-game win probability.")
    parser.add_argument("--home", required=True, help="Home team abbreviation, e.g. BOS")
    parser.add_argument("--away", required=True, help="Away team abbreviation, e.g. LAL")
    parser.add_argument("--home-rest", type=int, default=DEFAULT_REST_DAYS, help="Home rest days")
    parser.add_argument("--away-rest", type=int, default=DEFAULT_REST_DAYS, help="Away rest days")
    parser.add_argument("--season", type=int, default=None, help="Form season (default: latest)")
    args = parser.parse_args()

    home, away = args.home.upper(), args.away.upper()
    prob, ctx = predict(home, away, args.home_rest, args.away_rest, args.season)

    winner, win_prob = (home, prob) if prob >= 0.5 else (away, 1 - prob)
    season_label = f"{ctx['season']}-{(ctx['season'] + 1) % 100:02d}"
    print(f"\n{home} (home) vs {away} (away)  —  form from {season_label}")
    print(f"\n  {home}  {prob:6.1%}")
    print(f"  {away}  {1 - prob:6.1%}")
    print(f"\n  → {winner} favored ({win_prob:.1%})")

    print(f"\n  {'form (last 5/10 games)':<26}{home:>8}{away:>8}")
    for label, key in [
        ("win% last 5", "win_pct_5"),
        ("win% last 10", "win_pct_10"),
        ("pts scored", "pts_5"),
        ("pts allowed", "pts_allowed_5"),
        ("net rating", "net_rating_5"),
        ("rest days", "rest_days"),
    ]:
        print(f"  {label:<26}{ctx['home_feats'][key]:>8.2f}{ctx['away_feats'][key]:>8.2f}")


if __name__ == "__main__":
    main()