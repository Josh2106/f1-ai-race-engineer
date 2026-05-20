"""Feature engineering for the tyre-degradation model.

Input: clean per-lap DataFrame from `loader.get_lap_dataframe`.
Output: model-ready feature matrix with lap_time_s as the target.

Design choices (for your interview defense):
- We FILTER OUT pit-in/pit-out laps and Safety Car / VSC laps. Those laps
  have artificial lap times (slowing for pits, neutralized for SC) and would
  poison the tyre-degradation signal.
- We FUEL-CORRECT lap times. F1 cars burn ~1.7 kg/lap; ~0.035s/lap faster
  per kg lost. So a stint's later laps are inherently faster from fuel burn
  alone, NOT tyre improvement. We add a 'fuel_adjusted_lap_time' that
  removes this effect so the model can learn pure tyre behavior.
- We log the TRACK so the model can learn track-specific degradation rates
  (Monaco wears tyres very differently from Silverstone).
"""

from __future__ import annotations

import pandas as pd

# ~kg fuel burned per lap (rough average across tracks)
FUEL_KG_PER_LAP = 1.7
# Lap-time penalty per kg of fuel (rule of thumb used by engineers/sim work)
FUEL_TIME_PER_KG = 0.035


def add_fuel_correction(df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """Add a `fuel_adjusted_lap_time` column.

    Logic: at lap N of an L-lap race, remaining fuel mass is roughly
    (L - N) * FUEL_KG_PER_LAP. Convert this to seconds and add it to
    the lap time, so all laps are normalized to 'lap with zero fuel'.
    """
    df = df.copy()
    fuel_remaining_kg = (total_laps - df["lap_number"]).clip(lower=0) * FUEL_KG_PER_LAP
    fuel_penalty_s = fuel_remaining_kg * FUEL_TIME_PER_KG
    df["fuel_adjusted_lap_time"] = df["lap_time_s"] + fuel_penalty_s
    return df


def filter_representative_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop laps that don't reflect pure pace.

    Removes: pit-in / pit-out laps, laps with non-green track status,
    rows missing lap_time or compound, and outliers > 110% of median pace.
    """
    df = df.dropna(subset=["lap_time_s", "compound", "tyre_life"]).copy()
    df = df[~df["pit_in_lap"] & ~df["pit_out_lap"]]
    # TrackStatus '1' = all clear / green. Anything else = SC/VSC/yellow/red.
    df = df[df["track_status"].astype(str) == "1"]
    # Drop UNKNOWN/TEST compounds
    df = df[df["compound"].isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])]
    # Per-driver outlier filter: drop laps slower than 110% of driver's median
    median_pace = df.groupby("driver")["lap_time_s"].transform("median")
    df = df[df["lap_time_s"] < 1.10 * median_pace]
    return df.reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    total_laps: int,
    track_name: str,
    year: int,
) -> pd.DataFrame:
    """Take a cleaned lap DataFrame and produce a model-ready frame.

    Returned columns:
        Numeric: tyre_life, lap_number, year
        Categorical (kept as strings, encoded later): compound, team, track
        Target: fuel_adjusted_lap_time
    """
    df = filter_representative_laps(df)
    df = add_fuel_correction(df, total_laps=total_laps)
    df["track"] = track_name
    df["year"] = year

    feature_cols = [
        "tyre_life",
        "lap_number",
        "year",
        "compound",
        "team",
        "track",
        "fuel_adjusted_lap_time",
    ]
    return df[feature_cols].dropna().reset_index(drop=True)
