"""Train the pre-game win-probability model.

Splits are by SEASON, never random: a random split would put games from the
future in the training set, leaking information the model wouldn't have at
prediction time and inflating scores.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from courtvision.config import MODELS_DIR, PROCESSED_DIR
from courtvision.evaluation.metrics import (
    baseline_scores,
    calibration_table,
    evaluate,
    format_comparison,
)
from courtvision.features.build import FEATURE_COLS

# The model sees each pre-tipoff feature twice: once per team.
FEATURE_NAMES = [f"{col}_{side}" for side in ("home", "away") for col in FEATURE_COLS]

TARGET = "home_win"


def load_features():
    """Load the processed feature table, dropping rows with missing features."""
    path = PROCESSED_DIR / "game_features.csv"
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)


def split_by_season(df, train_end=2022, val_season=2023):
    """Chronological split: train on early seasons, validate, then test on the latest."""
    train = df[df["season"] <= train_end]
    val = df[df["season"] == val_season]
    test = df[df["season"] > val_season]
    return train, val, test


def build_model():
    """Scaled logistic regression.

    StandardScaler matters here: net_rating_5 spans ±31 while win_pct_5 is 0-1,
    and without scaling the larger-magnitude features dominate the coefficients.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train(df=None, train_end=2022, val_season=2023, save=True):
    """Fit the model and report performance against the home-team baseline."""
    df = load_features() if df is None else df
    train_df, val_df, test_df = split_by_season(df, train_end, val_season)

    model = build_model()
    model.fit(train_df[FEATURE_NAMES], train_df[TARGET])

    home_win_rate = train_df[TARGET].mean()

    print(f"train: {len(train_df):>6} games  (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    print(f"val:   {len(val_df):>6} games  (season {val_season})")
    print(f"test:  {len(test_df):>6} games  (seasons {test_df['season'].min()}-{test_df['season'].max()})")
    print(f"\ntrain home-win rate (baseline): {home_win_rate:.4f}")

    for name, split in (("VALIDATION", val_df), ("TEST", test_df)):
        probs = model.predict_proba(split[FEATURE_NAMES])[:, 1]
        print(f"\n=== {name} (season(s) {sorted(split['season'].unique())}) ===")
        print(format_comparison(evaluate(split[TARGET], probs), baseline_scores(split[TARGET], home_win_rate)))

    # Calibration on the test set: does "70%" actually mean 70%?
    test_probs = model.predict_proba(test_df[FEATURE_NAMES])[:, 1]
    print("\n=== TEST calibration ===")
    print(f"{'bin':<12}{'n':>6}{'predicted':>12}{'actual':>10}")
    for label, n, pred, actual in calibration_table(test_df[TARGET], test_probs):
        print(f"{label:<12}{n:>6}{pred:>12.3f}{actual:>10.3f}")

    print("\n=== feature coefficients (scaled) ===")
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_NAMES)
    print(coefs.sort_values(key=abs, ascending=False).round(4).to_string())

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "logreg_pregame.joblib")
        print(f"\nsaved model → {MODELS_DIR / 'logreg_pregame.joblib'}")

    return model


if __name__ == "__main__":
    train()