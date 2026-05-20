"""Pit-stop strategy engine.

Given the current race state (driver, current lap, current tyre & age,
remaining laps, track) and the trained tyre model, brute-force-simulate
every reasonable strategy and return them ranked by projected total time.

This is intentionally simple: real F1 sims account for safety cars, traffic,
fuel-saving, undercut/overcut dynamics, etc. We start with the cleanest
possible version — pure tyre-pace projection — and iterate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from src.models.tyre_degradation import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    predict_lap_time,
)


def _batch_predict(model, rows: pd.DataFrame) -> list[float]:
    """Predict lap times for many rows in one call. Much faster than per-row."""
    df = rows.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return list(model.predict(df[NUMERIC_COLS + CATEGORICAL_COLS]))

# Track-level pit loss in seconds (time lost in pit lane vs. staying out).
# Real numbers from team data / public estimates. Defaults to 22s.
PIT_LOSS_S = {
    "Monaco": 19.0,
    "Singapore": 28.0,
    "Bahrain": 22.0,
    "Silverstone": 21.0,
    "Monza": 22.0,
    "Spain": 22.0,
    "Abu Dhabi": 21.0,
    "Brazil": 21.0,
}

COMPOUND_FRESH_LIFE = {"SOFT": 1.0, "MEDIUM": 1.0, "HARD": 1.0}


@dataclass
class StintPlan:
    """A single stint: starting lap, ending lap (exclusive), and compound."""
    start_lap: int
    end_lap: int          # inclusive of the last lap on this compound
    compound: str

    @property
    def length(self) -> int:
        return self.end_lap - self.start_lap + 1


@dataclass
class Strategy:
    """A complete race strategy: list of stints + projected total time."""
    stints: list[StintPlan]
    projected_total_s: float = 0.0
    per_lap_breakdown: list[float] = field(default_factory=list)

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def pit_laps(self) -> list[int]:
        return [s.end_lap for s in self.stints[:-1]]

    def summary(self) -> str:
        stint_str = " | ".join(
            f"{s.compound[:3]} L{s.start_lap}-{s.end_lap}" for s in self.stints
        )
        return (
            f"{self.n_stops}-stop  {stint_str}  "
            f"total={self.projected_total_s:.1f}s"
        )


def simulate_strategy(
    model,
    strategy: Strategy,
    *,
    team: str,
    track: str,
    year: int,
) -> Strategy:
    """Run a single strategy through the tyre model and fill in projected_total_s.

    Implementation: builds a single DataFrame for ALL laps in the strategy
    and runs ONE batched XGBoost prediction. ~100x faster than per-lap calls.
    """
    pit_loss = PIT_LOSS_S.get(track, 22.0)

    rows = []
    for stint in strategy.stints:
        for offset, lap in enumerate(range(stint.start_lap, stint.end_lap + 1)):
            tyre_age = offset + COMPOUND_FRESH_LIFE.get(stint.compound, 1.0)
            rows.append({
                "tyre_life": tyre_age,
                "lap_number": lap,
                "year": year,
                "compound": stint.compound,
                "team": team,
                "track": track,
            })

    preds = _batch_predict(model, pd.DataFrame(rows))
    total = float(sum(preds)) + pit_loss * (len(strategy.stints) - 1)

    strategy.projected_total_s = total
    strategy.per_lap_breakdown = preds
    return strategy


def enumerate_one_stop(
    total_laps: int,
    starting_compound: str,
    candidate_pit_laps: Iterable[int],
    candidate_compounds: Iterable[str],
) -> list[Strategy]:
    """All 1-stop strategies: pit on each candidate lap, switch to each candidate compound."""
    out = []
    for pit_lap in candidate_pit_laps:
        if pit_lap >= total_laps or pit_lap < 2:
            continue
        for new_compound in candidate_compounds:
            if new_compound == starting_compound:
                continue
            out.append(Strategy(stints=[
                StintPlan(1, pit_lap, starting_compound),
                StintPlan(pit_lap + 1, total_laps, new_compound),
            ]))
    return out


def enumerate_two_stop(
    total_laps: int,
    starting_compound: str,
    candidate_pit_laps: Iterable[int],
    candidate_compounds: Iterable[str],
    *,
    min_stint_length: int = 8,
) -> list[Strategy]:
    """All 2-stop strategies with reasonable stint lengths."""
    out = []
    candidate_pit_laps = sorted(candidate_pit_laps)
    for pit1 in candidate_pit_laps:
        if pit1 < min_stint_length or pit1 >= total_laps - 2 * min_stint_length:
            continue
        for pit2 in candidate_pit_laps:
            if pit2 < pit1 + min_stint_length or pit2 >= total_laps - min_stint_length:
                continue
            for c2 in candidate_compounds:
                if c2 == starting_compound:
                    continue
                for c3 in candidate_compounds:
                    if c3 == c2:  # don't pit for the same compound back-to-back
                        continue
                    out.append(Strategy(stints=[
                        StintPlan(1, pit1, starting_compound),
                        StintPlan(pit1 + 1, pit2, c2),
                        StintPlan(pit2 + 1, total_laps, c3),
                    ]))
    return out


def recommend(
    model,
    *,
    team: str,
    track: str,
    year: int,
    total_laps: int,
    starting_compound: str = "MEDIUM",
    candidate_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    pit_window_spacing: int = 2,
    top_k: int = 5,
) -> list[Strategy]:
    """Brute-force every reasonable 1-stop and 2-stop strategy, return top_k by projected time.

    Real teams also consider 3-stop on high-deg tracks; that's a future extension.
    """
    candidate_pit_laps = list(range(10, total_laps - 5, pit_window_spacing))

    strategies: list[Strategy] = []
    strategies += enumerate_one_stop(total_laps, starting_compound,
                                     candidate_pit_laps, candidate_compounds)
    strategies += enumerate_two_stop(total_laps, starting_compound,
                                     candidate_pit_laps, candidate_compounds)

    for s in strategies:
        simulate_strategy(model, s, team=team, track=track, year=year)

    strategies.sort(key=lambda s: s.projected_total_s)
    return strategies[:top_k]


def _to_dataframe(strategies: list[Strategy]) -> pd.DataFrame:
    """Convenience: turn a list of strategies into a clean ranking DataFrame."""
    rows = []
    for rank, s in enumerate(strategies, 1):
        rows.append({
            "rank": rank,
            "stops": s.n_stops,
            "pit_laps": s.pit_laps,
            "compounds": [stint.compound for stint in s.stints],
            "total_s": round(s.projected_total_s, 2),
        })
    return pd.DataFrame(rows)
