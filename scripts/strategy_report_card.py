"""Week 3 deliverable: run the strategy engine on real historical races and
compare its recommendations to what the team actually did.

Usage:
    python scripts/strategy_report_card.py

Output: prints a side-by-side comparison for several past races.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import get_lap_dataframe, list_pit_stops, load_race
from src.models.pit_strategy import _to_dataframe, recommend
from src.models.tyre_degradation import load

# Races to evaluate. Each entry: (year, gp, driver_abbrev, starting_compound)
EVAL_RACES = [
    (2023, "Bahrain", "VER", "SOFT"),
    (2023, "Silverstone", "HAM", "MEDIUM"),
    (2023, "Monza", "VER", "MEDIUM"),
    (2024, "Monaco", "LEC", "MEDIUM"),
    (2024, "Silverstone", "HAM", "SOFT"),
]


def main() -> None:
    print("Loading trained tyre model...")
    model = load()

    for year, gp, driver, start_compound in EVAL_RACES:
        print("\n" + "=" * 70)
        print(f"  {year} {gp} GP — {driver}, starting on {start_compound}")
        print("=" * 70)

        try:
            sess = load_race(year, gp, "R")
            df = get_lap_dataframe(sess)
            total_laps = int(df["lap_number"].max())

            # Find the driver's team
            driver_team = (
                df[df["driver"] == driver]["team"].dropna().iloc[0]
                if not df[df["driver"] == driver].empty else "Mercedes"
            )

            # What the team actually did
            pits = list_pit_stops(df)
            actual_pits = pits[pits["driver"] == driver]
            print(f"\n  Actual race ({driver_team}, {total_laps} laps):")
            if actual_pits.empty:
                print("    No pit stops (likely a wet/short race)")
            else:
                for _, row in actual_pits.iterrows():
                    print(f"    Pit lap {int(row['lap_number'])} after "
                          f"{int(row['stint_length'])} laps on {row['compound_in']}")

            # What the AI recommends
            top = recommend(
                model,
                team=driver_team,
                track=gp,
                year=year,
                total_laps=total_laps,
                starting_compound=start_compound,
                top_k=5,
            )
            print(f"\n  AI top 5 strategies:")
            for s in top:
                print(f"    {s.summary()}")

        except Exception as exc:
            print(f"  !! Error: {exc}")


if __name__ == "__main__":
    main()
