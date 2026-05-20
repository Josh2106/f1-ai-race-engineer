"""Race replay: step through a historical race lap-by-lap.

For each lap we produce a `RaceState` snapshot (standings, gaps, recent events)
that downstream UI + commentary can consume.

The 'event detector' compares consecutive laps and emits race events when
something interesting changes (pit stop, lead change, fastest lap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

from src.commentary.prompt_templates import RaceEvent


@dataclass
class RaceState:
    """Snapshot of the race at the end of a given lap."""
    lap: int
    total_laps: int
    standings: list[tuple[int, str, str, float]]  # (pos, driver, team, gap_to_leader_s)
    events: list[RaceEvent] = field(default_factory=list)


def _gap_to_leader_seconds(lap_df: pd.DataFrame) -> dict[str, float]:
    """Approximate gap-to-leader by summing each driver's lap_time differences.

    Crude but effective for v1: real F1 timing uses precise sector-level
    accumulation. Good enough for the leaderboard display.
    """
    leader_total = lap_df["cumulative_time_s"].min()
    return (lap_df.set_index("driver")["cumulative_time_s"] - leader_total).to_dict()


def _build_standings_for_lap(df_to_lap: pd.DataFrame) -> list[tuple[int, str, str, float]]:
    """Given all laps up to and including `lap`, return current standings."""
    # latest row per driver
    latest = (
        df_to_lap.sort_values("lap_number")
        .groupby("driver", as_index=False)
        .last()
    )
    # cumulative time per driver
    cumtime = (
        df_to_lap.dropna(subset=["lap_time_s"])
        .groupby("driver")["lap_time_s"]
        .sum()
        .rename("cumulative_time_s")
        .reset_index()
    )
    merged = latest.merge(cumtime, on="driver", how="left")
    merged = merged.dropna(subset=["cumulative_time_s"])
    merged = merged.sort_values("cumulative_time_s")

    leader = merged["cumulative_time_s"].iloc[0]
    standings = []
    for pos, (_, row) in enumerate(merged.iterrows(), 1):
        gap = float(row["cumulative_time_s"] - leader)
        standings.append((pos, row["driver"], row["team"], gap))
    return standings


def detect_events(prev_lap_df: pd.DataFrame, this_lap_df: pd.DataFrame,
                  lap_number: int) -> list[RaceEvent]:
    """Compare two consecutive single-lap slices to spot interesting events."""
    events: list[RaceEvent] = []

    # Pit stops
    pit_drivers = this_lap_df[this_lap_df["pit_in_lap"]]["driver"].tolist()
    for drv in pit_drivers:
        compound = this_lap_df.loc[this_lap_df["driver"] == drv, "compound"].iloc[0]
        events.append(RaceEvent(
            type="pit_stop", lap=lap_number, driver=drv,
            detail=f"came in for fresh tyres, currently on {compound}",
        ))

    # Fastest lap (purple) - someone set a new fastest lap
    valid = this_lap_df.dropna(subset=["lap_time_s"])
    if not valid.empty:
        fastest = valid.loc[valid["lap_time_s"].idxmin()]
        # Was it a session-best? Approximate: compare against prev best
        if prev_lap_df is not None and not prev_lap_df.dropna(subset=["lap_time_s"]).empty:
            prev_best = prev_lap_df["lap_time_s"].min()
            if fastest["lap_time_s"] < prev_best:
                events.append(RaceEvent(
                    type="fastest_lap", lap=lap_number, driver=fastest["driver"],
                    detail=f"{fastest['lap_time_s']:.3f}s",
                ))

    return events


def iter_race_laps(df: pd.DataFrame, total_laps: int) -> Iterator[RaceState]:
    """Yield one RaceState per lap, including detected events."""
    # Precompute per-driver cumulative time once (much faster than per-lap)
    df = df.copy()
    df = df.sort_values(["driver", "lap_number"])
    df["lap_time_filled"] = df["lap_time_s"].fillna(df["lap_time_s"].median())
    df["cumulative_time_s"] = df.groupby("driver")["lap_time_filled"].cumsum()

    for lap in range(1, total_laps + 1):
        through_lap = df[df["lap_number"] <= lap]
        if through_lap.empty:
            continue

        this_lap_slice = df[df["lap_number"] == lap]
        prev_lap_slice = df[df["lap_number"] < lap] if lap > 1 else None
        events = detect_events(prev_lap_slice, this_lap_slice, lap)

        standings = _build_standings_for_lap(through_lap)
        yield RaceState(lap=lap, total_laps=total_laps, standings=standings, events=events)
