"""Week 4 deliverable: smoke-test the commentary engine on canned race events.

Generates Crofty-style commentary for 5 scripted F1 moments to verify:
  1. API connection works
  2. Persona stays in voice
  3. Streaming output renders smoothly

Usage:
    python scripts/commentary_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.commentary import CommentaryEngine, RaceEvent


# A snapshot of the 2024 Brazilian GP standings around lap 40 (rough)
STANDINGS = [
    (1, "VER", "Red Bull Racing", 0.0),
    (2, "OCO", "Alpine", 4.8),
    (3, "GAS", "Alpine", 8.2),
    (4, "RUS", "Mercedes", 11.5),
    (5, "LEC", "Ferrari", 14.1),
    (6, "HAM", "Mercedes", 18.7),
    (7, "SAI", "Ferrari", 22.4),
    (8, "NOR", "McLaren", 26.9),
]

EVENTS = [
    RaceEvent(
        type="race_start",
        lap=1,
        detail="lights out at a wet Interlagos; Verstappen started P17 after qualifying chaos",
    ),
    RaceEvent(
        type="overtake",
        lap=35,
        driver="VER",
        other_driver="GAS",
        detail="around the outside at Descida do Lago in the rain — VER takes P2",
    ),
    RaceEvent(
        type="pit_stop",
        lap=40,
        driver="VER",
        detail="2.5s stop, switched from INTERMEDIATE to HARD as track dries",
    ),
    RaceEvent(
        type="fastest_lap",
        lap=58,
        driver="VER",
        detail="purple sector splits, 1:24.118 — 0.6s faster than anyone else",
    ),
    RaceEvent(
        type="checkered_flag",
        lap=71,
        driver="VER",
        detail="wins from P17 in mixed conditions — one of the great drives",
    ),
]


def main() -> None:
    print("Initializing commentary engine (Groq + Llama 3.3 70B)...\n")
    engine = CommentaryEngine()

    for i, event in enumerate(EVENTS, 1):
        print(f"\n━━ Event {i}/{len(EVENTS)}: {event.type.upper()} on lap {event.lap} ━━")
        print()

        for chunk in engine.stream(
            event,
            track="Brazil",
            year=2024,
            total_laps=71,
            current_lap=event.lap,
            standings=STANDINGS,
        ):
            print(chunk, end="", flush=True)
            time.sleep(0.005)  # tiny delay so it feels like live commentary
        print("\n")

    print("Commentary smoke test complete.")


if __name__ == "__main__":
    main()
