"""Pre-render a complete race + AI commentary + strategy to a JSON file
for the static web/index.html viewer to play back.

Usage:
    python scripts/build_static_race.py --year 2024 --gp Brazil --driver VER

Output:
    web/data/race_<year>_<gp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.commentary import CommentaryEngine
from src.data.loader import get_lap_dataframe, load_race
from src.models.pit_strategy import recommend
from src.models.tyre_degradation import load as load_tyre_model
from src.replay import iter_race_laps

WEB_DATA = Path(__file__).resolve().parents[1] / "web" / "data"
WEB_DATA.mkdir(parents=True, exist_ok=True)


def _focus_info(df, driver: str):
    """Resolve team + starting compound for the focus driver."""
    team = df[df["driver"] == driver]["team"].dropna().iloc[0]
    start_compound = (
        df[(df["driver"] == driver) & (df["lap_number"] == 1)]["compound"]
        .dropna().iloc[0]
    )
    return team, start_compound


def build(year: int, gp: str, driver: str, *, max_commentary_events: int = 25) -> dict:
    print(f"Loading {year} {gp} GP...")
    sess = load_race(year, gp, "R")
    df = get_lap_dataframe(sess)
    total_laps = int(df["lap_number"].max())

    team, start_compound = _focus_info(df, driver)
    print(f"Focus driver: {driver} ({team}), starting on {start_compound}")

    print("Loading tyre model + running strategy engine...")
    model = load_tyre_model()
    top_strats = recommend(
        model, team=team, track=gp, year=year,
        total_laps=total_laps, starting_compound=start_compound, top_k=3,
    )
    strategy_payload = [
        {"rank": i + 1, "summary": s.summary(),
         "stops": s.n_stops, "total_s": round(s.projected_total_s, 1)}
        for i, s in enumerate(top_strats)
    ]

    print("Initializing commentary engine...")
    engine = CommentaryEngine()

    print(f"Walking race ({total_laps} laps), generating commentary on events...")
    laps_payload = []
    commentary_calls = 0

    for state in iter_race_laps(df, total_laps):
        events_payload = []
        for event in state.events:
            if commentary_calls >= max_commentary_events:
                break
            try:
                text = engine.say(
                    event,
                    track=gp, year=year, total_laps=total_laps,
                    current_lap=state.lap, standings=state.standings,
                )
                events_payload.append({
                    "type": event.type,
                    "driver": event.driver,
                    "text": text,
                })
                commentary_calls += 1
                print(f"  Lap {state.lap:3d} [{event.type:11s}] -> {text[:60]}...")
            except Exception as exc:
                print(f"  Lap {state.lap} commentary failed: {exc}")

        laps_payload.append({
            "lap": state.lap,
            "standings": [
                {"pos": pos, "driver": drv, "team": tm, "gap": round(gap, 2)}
                for pos, drv, tm, gap in state.standings[:10]
            ],
            "events": events_payload,
        })

    payload = {
        "meta": {
            "year": year, "gp": gp, "total_laps": total_laps,
            "focus_driver": driver, "focus_team": team,
            "starting_compound": start_compound,
            "commentary_events": commentary_calls,
        },
        "strategy": strategy_payload,
        "laps": laps_payload,
    }
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--gp", default="Brazil")
    p.add_argument("--driver", default="VER")
    p.add_argument("--max-events", type=int, default=25,
                   help="Cap on Groq calls to stay within rate limits")
    args = p.parse_args()

    payload = build(args.year, args.gp, args.driver,
                    max_commentary_events=args.max_events)

    fname = f"race_{args.year}_{args.gp.lower().replace(' ', '_')}.json"
    out_path = WEB_DATA / fname
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"Used {payload['meta']['commentary_events']} commentary calls.")


if __name__ == "__main__":
    main()
