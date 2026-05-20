"""LLM commentary generator using Groq's free Llama 3.3 70B endpoint.

Usage:
    engine = CommentaryEngine()
    text = engine.say(RaceEvent(type="pit_stop", lap=22, driver="VER",
                                 detail="2.3s stop, switched to HARD"),
                       race_context=...)
    # or for streaming:
    for chunk in engine.stream(...):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq

from .prompt_templates import CROFTY_SYSTEM_PROMPT, RaceEvent, format_race_state

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class CommentaryEngine:
    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or key.startswith("your_"):
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "and put it in .env (see .env.example)."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def _build_messages(
        self,
        event: RaceEvent,
        *,
        track: str,
        year: int,
        total_laps: int,
        current_lap: int,
        standings: list[tuple[int, str, str, float]],
        ai_strategy_note: str | None = None,
    ) -> list[dict]:
        user_text = format_race_state(
            track=track,
            year=year,
            total_laps=total_laps,
            current_lap=current_lap,
            standings=standings,
            event=event,
            ai_strategy_note=ai_strategy_note,
        )
        return [
            {"role": "system", "content": CROFTY_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def say(self, event: RaceEvent, **race_ctx) -> str:
        """One-shot commentary (non-streaming). Returns the full text."""
        messages = self._build_messages(event, **race_ctx)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()

    def stream(self, event: RaceEvent, **race_ctx) -> Iterator[str]:
        """Streaming commentary. Yields text chunks as they're generated."""
        messages = self._build_messages(event, **race_ctx)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=180,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
