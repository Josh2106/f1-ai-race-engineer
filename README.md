# F1 AI Race Engineer

> Replays any historical Formula 1 Grand Prix (2018+) with a real-time AI race engineer that recommends pit-stop strategy and an LLM that generates Crofty-style live commentary.

**Status:** Week 1 / 6 — Foundation

## What it does

- **Race Replay:** Pick any GP from 2018 onwards (FastF1 data) and watch it back at variable speed
- **AI Strategy:** XGBoost tyre-degradation model + Monte Carlo simulation recommends optimal pit windows
- **Live Commentary:** Llama 3.3 70B (via Groq) generates real-time race commentary in the style of David Croft
- **Visual Track Map:** Cars move around the circuit live with telemetry overlays

## Tech Stack

| Layer | Tool |
|---|---|
| Data | FastF1 |
| ML | XGBoost, scikit-learn |
| LLM | Groq (Llama 3.3 70B) |
| UI | Streamlit + Plotly |
| Optional TTS | edge-tts |

## Setup

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd f1-ai-race-engineer

# 2. Create virtual env (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install deps
pip install -r requirements.txt

# 4. Add Groq API key (get free key at console.groq.com)
copy .env.example .env
# Then edit .env and paste your key

# 5. Smoke test — load the 2023 Monaco GP
python scripts/smoke_test.py
```

## Project Structure

```
src/
  data/loader.py          # FastF1 wrappers, cleaned per-lap DataFrame
  data/features.py        # Feature engineering for ML model
  models/tyre_degradation.py  # XGBoost lap-time predictor
  models/pit_strategy.py  # Monte Carlo strategy simulator
  commentary/engine.py    # Groq LLM commentary generator
  commentary/prompt_templates.py  # Crofty persona
  replay/race_clock.py    # Lap-by-lap event stream
  app.py                  # Streamlit UI
notebooks/                # Exploration + model training
scripts/                  # CLI utilities (smoke tests, training)
```

## Roadmap

- [x] Project scaffold + FastF1 loader
- [ ] Tyre degradation model (Week 2)
- [ ] Strategy engine (Week 3)
- [ ] LLM commentary (Week 4)
- [ ] Streamlit app (Week 5)
- [ ] Demo video + polish (Week 6)

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) for free, high-quality F1 telemetry data
- [Groq](https://groq.com) for free fast LLM inference
