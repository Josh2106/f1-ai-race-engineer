"""The 'soul' of the commentary engine.

The system prompt defines persona. The format_race_state() helper turns
a structured race snapshot into compact text the LLM can reason about.

Design notes for interview defense:
- We give the LLM STRUCTURED race state, not raw telemetry, because
  language models are better at reasoning over a clean text summary.
- We constrain length (2-3 sentences) because real commentary is punchy.
- We tell the model what just happened ('event') AND broader context
  (positions, gaps) so it can place the event in the race narrative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CROFTY_SYSTEM_PROMPT = """You are David "Crofty" Croft, the lead Formula 1 commentator
for Sky Sports F1. You are commentating live on a Grand Prix.

Your style:
- High-energy, dramatic, but never hyperbolic to the point of cringe
- British, conversational, uses driver surnames ("Verstappen", "Hamilton")
- Knows F1 deeply: refers to teams ("the Mercedes"), tyre compounds
  ("the softs", "those mediums"), and race craft (undercut, dirty air,
  DRS train, etc.) naturally
- Reacts to drama: pit stops, overtakes, fastest laps, safety cars
- Short, punchy sentences. Aim for 2-3 sentences per response.

NEVER:
- Invent facts not in the race state (no fictional crashes, fake quotes)
- Use emojis or markdown formatting
- Speak in first person about yourself
- Start with "Ladies and gentlemen" or other studio-presenter cliches
- Exceed 3 sentences

Output ONLY the commentary text. No labels, no quotes, no narration."""


@dataclass
class RaceEvent:
    """A discrete thing that just happened in the race."""
    type: Literal[
        "pit_stop", "overtake", "fastest_lap", "safety_car",
        "yellow_flag", "race_start", "checkered_flag", "dnf", "tyre_alert",
        "general",
    ]
    lap: int
    driver: str | None = None         # primary driver involved
    other_driver: str | None = None   # for overtakes
    detail: str = ""                  # free-form extra info


def format_race_state(
    *,
    track: str,
    year: int,
    total_laps: int,
    current_lap: int,
    standings: list[tuple[int, str, str, float]],  # (pos, driver, team, gap_to_leader_s)
    event: RaceEvent,
    ai_strategy_note: str | None = None,
) -> str:
    """Compact text summary fed to the LLM as the USER message."""
    lines = [
        f"Race: {year} {track} Grand Prix — Lap {current_lap} of {total_laps}",
        "",
        "Standings (top 8):",
    ]
    for pos, drv, team, gap in standings[:8]:
        gap_str = "LEADER" if pos == 1 else f"+{gap:.1f}s"
        lines.append(f"  P{pos} {drv} ({team})  {gap_str}")

    lines.append("")
    lines.append("Event just now:")
    lines.append(f"  Type: {event.type}")
    if event.driver:
        lines.append(f"  Driver: {event.driver}")
    if event.other_driver:
        lines.append(f"  Other driver: {event.other_driver}")
    if event.detail:
        lines.append(f"  Detail: {event.detail}")

    if ai_strategy_note:
        lines.append("")
        lines.append(f"AI strategy note (for your awareness, don't quote verbatim): {ai_strategy_note}")

    lines.append("")
    lines.append("Now commentate on what just happened in 2-3 sentences, in your voice.")
    return "\n".join(lines)
