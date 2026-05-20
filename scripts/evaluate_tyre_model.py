"""Week 2 evaluation: visualize what the tyre model learned.

Generates 3 plots saved to models/figures/:
  1. degradation_curves.png  - lap time vs tyre age, per compound
  2. predicted_vs_actual.png - scatter on test set, perfect-line overlay
  3. residuals_by_track.png  - box plot showing per-track prediction error

Usage:
    python scripts/evaluate_tyre_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import build_feature_matrix
from src.data.loader import get_lap_dataframe, load_race
from src.models.tyre_degradation import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    load,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "models" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Held-out race the model NEVER saw during training
HOLDOUT_RACE = (2024, "Monaco")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("category")
    return out


def plot_degradation_curves(model) -> None:
    """For each compound at a representative track, plot predicted lap time
    as the tyre ages from 1 to 35 laps. This shows what the model 'thinks'
    each compound's degradation behavior is.
    """
    compounds = ["SOFT", "MEDIUM", "HARD"]
    ages = np.arange(1, 36)
    track = "Silverstone"     # high-deg track makes the curves clearly different
    team = "Mercedes"
    year = 2024

    fig, ax = plt.subplots(figsize=(9, 5))
    for compound in compounds:
        rows = pd.DataFrame({
            "tyre_life": ages,
            "lap_number": ages,
            "year": year,
            "compound": compound,
            "team": team,
            "track": track,
        })
        rows = _prep(rows)
        preds = model.predict(rows[NUMERIC_COLS + CATEGORICAL_COLS])
        ax.plot(ages, preds, label=compound, linewidth=2.2)

    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Predicted fuel-adjusted lap time (s)")
    ax.set_title(f"What the model learned about tyres\n({track}, {team}, {year})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "degradation_curves.png", dpi=120)
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'degradation_curves.png'}")


def plot_predicted_vs_actual(model, holdout_df: pd.DataFrame) -> None:
    """Scatter test-set predictions vs ground truth. Closer to the diagonal = better."""
    X = _prep(holdout_df[NUMERIC_COLS + CATEGORICAL_COLS])
    y = holdout_df[TARGET_COL]
    yhat = model.predict(X)

    mae = np.mean(np.abs(y - yhat))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, yhat, alpha=0.3, s=14)
    lo, hi = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual lap time (s)")
    ax.set_ylabel("Predicted lap time (s)")
    ax.set_title(f"Predicted vs Actual on held-out {HOLDOUT_RACE[0]} {HOLDOUT_RACE[1]}\n"
                 f"MAE = {mae:.3f}s on {len(y)} laps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=120)
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'predicted_vs_actual.png'}  (MAE on holdout: {mae:.3f}s)")


def plot_residuals_by_compound(model, holdout_df: pd.DataFrame) -> None:
    X = _prep(holdout_df[NUMERIC_COLS + CATEGORICAL_COLS])
    y = holdout_df[TARGET_COL].values
    yhat = model.predict(X)
    res = y - yhat

    fig, ax = plt.subplots(figsize=(8, 5))
    compounds = sorted(holdout_df["compound"].unique())
    data = [res[holdout_df["compound"].values == c] for c in compounds]
    ax.boxplot(data, tick_labels=compounds)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Residual: actual - predicted (s)")
    ax.set_title(f"Prediction error by compound\n({HOLDOUT_RACE[0]} {HOLDOUT_RACE[1]} holdout)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals_by_compound.png", dpi=120)
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'residuals_by_compound.png'}")


def main() -> None:
    print("Loading trained model...")
    model = load()

    print(f"Loading held-out race: {HOLDOUT_RACE[0]} {HOLDOUT_RACE[1]}...")
    sess = load_race(HOLDOUT_RACE[0], HOLDOUT_RACE[1], "R")
    raw = get_lap_dataframe(sess)
    holdout = build_feature_matrix(
        raw,
        total_laps=int(raw["lap_number"].max()),
        track_name=HOLDOUT_RACE[1],
        year=HOLDOUT_RACE[0],
    )
    print(f"  {len(holdout)} usable laps on holdout")

    print("\nGenerating plots:")
    plot_degradation_curves(model)
    plot_predicted_vs_actual(model, holdout)
    plot_residuals_by_compound(model, holdout)

    print(f"\nDone. Open {FIG_DIR} to see the plots.")


if __name__ == "__main__":
    main()
