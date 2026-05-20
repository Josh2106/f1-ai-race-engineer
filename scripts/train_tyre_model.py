"""Week 2 deliverable: train the tyre-degradation model on multiple races.

Usage:
    python scripts/train_tyre_model.py

This script:
  1. Loads several historical races (mixed tracks, ~3 years of data)
  2. Cleans + feature-engineers them via src.data.features
  3. Trains XGBoost and prints MAE / R^2
  4. Saves model to models/tyre_model.json

Expected runtime: 5-10 minutes on first run (FastF1 has to download each
race), <30s on subsequent runs (everything cached).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import build_feature_matrix
from src.data.loader import get_lap_dataframe, load_race
from src.models.tyre_degradation import save, train

# Mix of tracks (street/high-speed/mixed) and years (2022-2024) for generalization
TRAINING_RACES = [
    # (year, gp_name)
    (2022, "Bahrain"),
    (2022, "Spain"),
    (2022, "Monza"),
    (2023, "Bahrain"),
    (2023, "Monaco"),
    (2023, "Silverstone"),
    (2023, "Monza"),
    (2023, "Singapore"),
    (2024, "Bahrain"),
    (2024, "Spain"),
    (2024, "Silverstone"),
    (2024, "Monza"),
]


def load_all() -> pd.DataFrame:
    frames = []
    for year, gp in TRAINING_RACES:
        try:
            print(f"  Loading {year} {gp}...", flush=True)
            sess = load_race(year, gp, "R")
            raw = get_lap_dataframe(sess)
            feat = build_feature_matrix(
                raw,
                total_laps=int(raw["lap_number"].max()),
                track_name=gp,
                year=year,
            )
            print(f"    -> {len(feat)} usable laps")
            frames.append(feat)
        except Exception as exc:
            print(f"    !! skipped: {exc}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print(f"Loading {len(TRAINING_RACES)} races...")
    df = load_all()
    print(f"\nTotal training rows: {len(df)}")
    print(f"Tracks: {sorted(df['track'].unique())}")
    print(f"Compounds: {sorted(df['compound'].unique())}")
    print(f"Teams: {len(df['team'].unique())}\n")

    print("Training XGBoost...")
    model, report = train(df)
    print(report.summary())

    save(model, metadata={
        "races": [f"{y} {g}" for y, g in TRAINING_RACES],
        "n_rows": len(df),
        "mae_test": report.mae_test,
        "r2_test": report.r2_test,
    })
    print(f"\nSaved model -> models/tyre_model.json")
    print("\nTarget acceptance: Test MAE < 0.8s. Stretch: < 0.5s.")


if __name__ == "__main__":
    main()
