# CourtVision — NBA Win Probability Engine

A production-style system that predicts NBA game outcomes before tip-off and updates
win probability live as the game unfolds — served through a real API and a browsable
dashboard, not just a notebook.

## Project layout

```
CourtVision/
├── src/courtvision/        # Python package (src layout)
│   ├── config.py           # paths + env settings — import from here
│   ├── data/               # acquisition/ingestion (nba_api → data/raw/)
│   ├── features/           # feature engineering (leakage-safe, pre-tipoff only)
│   ├── models/             # training, calibration, persistence
│   └── evaluation/         # metrics, baselines, calibration curves
├── data/                   # raw / interim / processed (gitignored, .gitkeep tracked)
├── models/                 # trained artifacts (gitignored)
├── notebooks/              # exploration
├── tests/                  # pytest
├── pyproject.toml          # deps + tooling; phase deps under optional-dependencies
├── Makefile                # setup / test / lint / format
└── .env.example            # copy to .env
```

## Getting started

```bash
make setup                  # creates .venv and installs the package + dev tools
source .venv/bin/activate
cp .env.example .env
make test                   # smoke tests should pass
```

Dependencies are staged by phase. Phase 1 runtime deps install by default; later phases
pull extras: `pip install -e ".[modeling]"` (Phase 3), `pip install -e ".[api]"` (Phase 2).

## Roadmap

- **Phase 1** — Pre-game predictor: pull historical data (nba_api), build leakage-safe
  features, train a logistic-regression baseline, beat the "home team always wins" (~59%) baseline.
- **Phase 2** — Full-stack MVP: FastAPI backend, PostgreSQL, React dashboard.
- **Phase 3** — Model depth: XGBoost/LightGBM, player features, MLflow tracking.
- **Phase 4** — Live win probability: in-game state (score, time, possession).
- **Phase 5** — Productionize: Docker, public deploy, daily refresh via GitHub Actions.

## ML principles this project practices

Season-based train/val/test splits (no random splits), data-leakage guards, baseline
comparison, and **probability calibration** — a "70%" prediction should be right ~70%
of the time, measured by Brier score and calibration curves.
