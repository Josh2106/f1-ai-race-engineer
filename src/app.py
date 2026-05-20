"""F1 AI Race Engineer — Streamlit app.

Run:
    streamlit run src/app.py

Pick a race, hit play, watch the AI commentate and recommend strategy live.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.commentary import CommentaryEngine
from src.commentary.prompt_templates import RaceEvent
from src.data.loader import get_lap_dataframe, load_race
from src.models.pit_strategy import recommend
from src.models.tyre_degradation import load as load_tyre_model
from src.replay import iter_race_laps

st.set_page_config(page_title="F1 AI Race Engineer", page_icon="🏎️", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# Sidebar — race & playback controls
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🏎️ F1 AI Race Engineer")
st.sidebar.markdown("Replay any GP with an AI race engineer + live commentary.")

YEARS = [2022, 2023, 2024]
GP_OPTIONS = {
    2022: ["Bahrain", "Spain", "Monza", "Brazil"],
    2023: ["Bahrain", "Monaco", "Silverstone", "Monza", "Singapore"],
    2024: ["Bahrain", "Monaco", "Spain", "Silverstone", "Monza", "Brazil"],
}

year = st.sidebar.selectbox("Year", YEARS, index=2)
gp = st.sidebar.selectbox("Grand Prix", GP_OPTIONS[year], index=len(GP_OPTIONS[year]) - 1)

speed = st.sidebar.selectbox("Speed", ["0.5s/lap", "1s/lap", "2s/lap", "5s/lap"], index=1)
speed_s = float(speed.split("s")[0])

commentary_on = st.sidebar.checkbox("AI commentary (Groq)", value=True)
strategy_driver = st.sidebar.text_input("Strategy focus driver", value="VER")

start_btn = st.sidebar.button("▶ Start race replay", type="primary", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Free LLM via Groq. Data from FastF1.")

# ─────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def _tyre_model():
    return load_tyre_model()


@st.cache_data(show_spinner=False)
def _race_data(year: int, gp: str):
    sess = load_race(year, gp, "R")
    return get_lap_dataframe(sess)


@st.cache_resource
def _commentary_engine():
    return CommentaryEngine()


# ─────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"## {year} {gp} Grand Prix")

left, right = st.columns([1.2, 1])

with left:
    lap_indicator = st.empty()
    chart_placeholder = st.empty()
    standings_placeholder = st.empty()

with right:
    strategy_placeholder = st.container()
    st.markdown("### 💬 Live Commentary")
    commentary_placeholder = st.container()

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def lap_time_chart(df: pd.DataFrame, drivers: list[str], up_to_lap: int) -> go.Figure:
    fig = go.Figure()
    sub = df[(df["lap_number"] <= up_to_lap) & df["driver"].isin(drivers)]
    sub = sub.dropna(subset=["lap_time_s"])
    for drv in drivers:
        d = sub[sub["driver"] == drv]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["lap_number"], y=d["lap_time_s"],
            mode="lines+markers", name=drv,
            line=dict(width=2),
        ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Lap", yaxis_title="Lap time (s)",
        legend=dict(orientation="h", y=-0.25),
        template="plotly_dark",
    )
    return fig


def render_standings(standings: list[tuple[int, str, str, float]]) -> pd.DataFrame:
    rows = [{
        "P": pos, "Driver": drv, "Team": team,
        "Gap": "LEADER" if pos == 1 else f"+{gap:.1f}s",
    } for pos, drv, team, gap in standings[:10]]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Run the race
# ─────────────────────────────────────────────────────────────────────
if start_btn:
    with st.spinner(f"Loading {year} {gp} GP data..."):
        df = _race_data(year, gp)
        total_laps = int(df["lap_number"].max())
        model = _tyre_model()

    # Get one-off strategy recommendation for the focus driver
    try:
        driver_team = df[df["driver"] == strategy_driver]["team"].dropna().iloc[0]
        start_compound = (
            df[(df["driver"] == strategy_driver) & (df["lap_number"] == 1)]["compound"]
            .dropna().iloc[0]
        )
        top_strats = recommend(
            model, team=driver_team, track=gp, year=year,
            total_laps=total_laps, starting_compound=start_compound, top_k=3,
        )
    except (IndexError, KeyError):
        top_strats = []
        driver_team = "?"
        start_compound = "?"

    with strategy_placeholder:
        st.markdown("### 🧠 AI Strategy")
        st.caption(f"For **{strategy_driver}** ({driver_team}), starting on {start_compound}")
        for i, s in enumerate(top_strats, 1):
            label = f"**#{i}** — {s.summary()}"
            if i == 1:
                st.success(label)
            else:
                st.info(label)

    # Commentary engine (lazy init to allow checkbox off)
    engine = _commentary_engine() if commentary_on else None

    # Find top 5 drivers by final position (cool to track on chart)
    final_standings = (
        df.dropna(subset=["lap_time_s"])
        .groupby("driver")["lap_time_s"].sum()
        .sort_values().head(5).index.tolist()
    )

    commentary_lines: list[str] = []
    for state in iter_race_laps(df, total_laps):
        lap_indicator.markdown(f"### Lap {state.lap} / {state.total_laps}")
        chart_placeholder.plotly_chart(
            lap_time_chart(df, final_standings, state.lap),
            use_container_width=True, key=f"chart_lap_{state.lap}",
        )
        standings_placeholder.dataframe(
            render_standings(state.standings),
            hide_index=True, use_container_width=True,
        )

        # Commentary on events
        if engine is not None and state.events:
            for event in state.events:
                try:
                    text = engine.say(
                        event,
                        track=gp, year=year, total_laps=total_laps,
                        current_lap=state.lap, standings=state.standings,
                    )
                    commentary_lines.append(f"**Lap {state.lap}** — {text}")
                except Exception as exc:
                    commentary_lines.append(f"_(commentary error: {exc})_")
            # Render the last 6 events to keep it readable
            with commentary_placeholder:
                commentary_placeholder.empty()
                for line in commentary_lines[-6:]:
                    st.markdown(line)
                    st.markdown("---")

        time.sleep(speed_s)

    st.balloons()
    st.success(f"Race complete. Total laps simulated: {total_laps}")
else:
    st.info("Pick a race in the sidebar and hit **Start race replay** to begin.")
