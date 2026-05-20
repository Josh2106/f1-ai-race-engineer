"""FastF1 wrappers: load any historical race and return a clean per-lap DataFrame.

A 'lap row' is the unit of analysis for every downstream model.
"""

from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_race(year: int, gp: str | int, session: str = "R"):
    """Load a session. `gp` accepts round number (1-24) or GP name ('Monaco').

    `session` is one of: 'R' (race), 'Q' (qualifying), 'S' (sprint),
    'FP1', 'FP2', 'FP3'.
    """
    sess = fastf1.get_session(year, gp, session)
    sess.load(laps=True, telemetry=False, weather=True, messages=True)
    return sess


def get_lap_dataframe(session) -> pd.DataFrame:
    """Return a tidy per-lap DataFrame with the columns we'll use everywhere.

    Columns:
        driver, team, lap_number, stint, compound, tyre_life,
        lap_time_s, sector_1_s, sector_2_s, sector_3_s,
        position, pit_in_lap, pit_out_lap, track_status, is_personal_best
    """
    laps = session.laps.copy()

    def _to_seconds(td):
        return td.total_seconds() if pd.notnull(td) else None

    df = pd.DataFrame({
        "driver": laps["Driver"],
        "team": laps["Team"],
        "lap_number": laps["LapNumber"].astype("Int64"),
        "stint": laps["Stint"].astype("Int64"),
        "compound": laps["Compound"],
        "tyre_life": laps["TyreLife"].astype("Float64"),
        "lap_time_s": laps["LapTime"].apply(_to_seconds),
        "sector_1_s": laps["Sector1Time"].apply(_to_seconds),
        "sector_2_s": laps["Sector2Time"].apply(_to_seconds),
        "sector_3_s": laps["Sector3Time"].apply(_to_seconds),
        "position": laps["Position"].astype("Float64"),
        "pit_in_lap": laps["PitInTime"].notna(),
        "pit_out_lap": laps["PitOutTime"].notna(),
        "track_status": laps["TrackStatus"],
        "is_personal_best": laps["IsPersonalBest"].fillna(False).astype(bool),
    }).reset_index(drop=True)

    return df


def list_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per pit stop: driver, lap, in/out compound."""
    pit_in = df[df["pit_in_lap"]].copy()
    return pit_in[["driver", "lap_number", "compound", "tyre_life"]].rename(
        columns={"compound": "compound_in", "tyre_life": "stint_length"}
    )
