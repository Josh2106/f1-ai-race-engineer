"""Week 1 smoke test: load 2023 Monaco GP and print a summary.

Run from the project root:
    python scripts/smoke_test.py

First run downloads + caches ~50 MB; subsequent runs are instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import get_lap_dataframe, list_pit_stops, load_race


def main() -> None:
    print("Loading 2023 Monaco GP race session... (first run takes ~30s)")
    sess = load_race(2023, "Monaco", "R")
    df = get_lap_dataframe(sess)

    print(f"\nLoaded {len(df)} laps across {df['driver'].nunique()} drivers.")
    print(f"Race winner: {sess.results.iloc[0]['Abbreviation']} "
          f"({sess.results.iloc[0]['TeamName']})")

    ver = df[df["driver"] == "VER"].dropna(subset=["lap_time_s"])
    if not ver.empty:
        print(f"\nVerstappen — {len(ver)} timed laps")
        print(f"  Fastest lap: {ver['lap_time_s'].min():.3f}s "
              f"on lap {int(ver.loc[ver['lap_time_s'].idxmin(), 'lap_number'])}")
        print(f"  Average:     {ver['lap_time_s'].mean():.3f}s")

    print("\nPit stops:")
    pits = list_pit_stops(df)
    print(pits.to_string(index=False))

    print("\nSmoke test passed. FastF1 cache:",
          Path(__file__).resolve().parents[1] / "data" / "cache")


if __name__ == "__main__":
    main()
