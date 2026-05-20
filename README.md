# 🏎️ F1 AI Race Engineer

> An end-to-end AI system that replays historical Formula 1 Grand Prix with a learned tyre-degradation model recommending pit-stop strategy and a large language model generating live broadcast-style commentary.

**🔴 Live demo:** *(coming soon — link will go here after GitHub Pages is enabled)*

---

## What it does

Pick a race. The site replays it lap-by-lap in your browser with three layers of AI working together:

1. **Race Replay** — historical telemetry from the FastF1 API streamed lap-by-lap with a live-updating gap-to-leader chart and standings table.
2. **AI Pit Strategy** — an XGBoost tyre-degradation model is wrapped in a brute-force Monte Carlo simulator that enumerates hundreds of pit-stop strategies and recommends the top three by projected race time.
3. **AI Live Commentary** — race events (pit stops, fastest laps) trigger calls to Llama 3.3 70B (via Groq) which narrates them in the voice of Sky Sports' David Croft.

The whole thing is **pre-rendered to a static JSON file** so the website needs no backend — hostable for free on GitHub Pages.

---

## Demo

*(Add a screenshot or GIF here once you've recorded one — call it `web/demo.png` and reference it as `![demo](web/demo.png)`)*

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       OFFLINE BUILD (Python)                      │
│                                                                   │
│   FastF1 API ──▶ src/data/loader.py ──▶ feature engineering      │
│                                                ▼                  │
│   pre-trained XGBoost model ◀── train_tyre_model.py              │
│         │                                                         │
│         ▼                                                         │
│   src/models/pit_strategy.py   (Monte Carlo over ~500 strategies)│
│         │                                                         │
│         ▼                                                         │
│   src/replay/race_clock.py     (per-lap event detection)         │
│         │                                                         │
│         ▼                                                         │
│   src/commentary/engine.py     (Groq → Llama 3.3 70B)            │
│         │                                                         │
│         ▼                                                         │
│   web/data/race_<year>_<gp>.json   (~100 KB per race)            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  RUNTIME (browser, no server)                     │
│   web/index.html  +  vanilla JS  +  Chart.js (CDN)                │
│   plays the JSON back lap-by-lap with animated UI                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key results

| Metric | Value | Notes |
|---|---|---|
| Tyre-degradation MAE (test) | **0.485 s/lap** | comparable to academic F1 modeling papers |
| Training data | 10,967 cleaned laps | 12 races × 3 years × mixed track types |
| Strategy accuracy | within **3 laps** of real team call on 2024 Silverstone | Hamilton 1-stop, AI recommended lap 24 vs actual 27 |
| Commentary latency | < 1.5 s/event | Groq's free Llama 3.3 70B endpoint |
| Site weight | ~150 KB per pre-rendered race | static hostable, no server cost |

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Data | **FastF1** | Free, official-quality F1 telemetry (2018+) |
| ML | **XGBoost** | Industry-standard gradient-boosted trees, native categorical support |
| Strategy sim | **NumPy + pandas (batched inference)** | ~100× speedup over per-lap predict calls |
| LLM | **Groq + Llama 3.3 70B** | Free tier, sub-second response, no GPU needed locally |
| Frontend | **Vanilla JS + Chart.js (CDN)** | No build step, no framework lock-in |
| Hosting | **GitHub Pages** | Free, zero ops |

---

## Methodology

### Feature engineering decisions

- **Fuel correction.** F1 cars burn ~1.7 kg/lap and each kg costs ~0.035 s/lap. Without correction, later stint laps look artificially faster — corrupting the tyre-aging signal. The model predicts a fuel-adjusted lap time so it can learn pure tyre behavior.
- **Filtering.** Pit-in/out laps, Safety Car laps, and laps slower than 110% of a driver's median are dropped — they reflect race context, not tyre pace.
- **Categorical encoding.** XGBoost's `enable_categorical=True` handles compound, team, and track natively (no one-hot blow-up).

### Honest limitations (and what I'd add next)

- The model **doesn't know about weather** — it can't predict the rain-induced strategy chaos at 2024 Silverstone or 2024 Brazil. Adding weather features (track temp, rainfall) would help.
- **Sparse data beyond ~25-lap stints** causes the model to extrapolate; the degradation curves show a "cliff" around lap 25. A parametric tyre model (compound-specific linear degradation) layered on top would be more robust.
- **R² = 0.99 is misleading** — `track` is a feature, so the model trivially learns Monaco ≈ 75s vs Monza ≈ 80s. The honest performance metric is the **0.485 s MAE**.
- Pit loss is a single track-level constant; in reality it varies with traffic and pit-lane queue.

### Why pre-render instead of live?

Trade-off explicitly considered. Pre-rendering means:
- ✅ **Zero runtime cost** (no Groq calls on page load, no backend server)
- ✅ **Reproducible demo** — the commentary doesn't change between viewers
- ✅ **Hostable on GitHub Pages**
- ❌ Each race build uses ~25 Groq calls and ~2 min — fine for portfolio, not for a live product

For a production app I'd swap in FastAPI + Redis + a websocket stream.

---

## Run it yourself

### Prereqs

- Python 3.11+
- A free **Groq API key** (https://console.groq.com)

### Setup

```bash
git clone https://github.com/<your-username>/f1-ai-race-engineer
cd f1-ai-race-engineer

python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt

cp .env.example .env             # then paste your GROQ_API_KEY
```

### Pipeline

```bash
# 1. Sanity-check the data loader
python scripts/smoke_test.py

# 2. Train the tyre-degradation model (~5-10 min first run, caches afterwards)
python scripts/train_tyre_model.py

# 3. Visualize what the model learned (saves plots to models/figures/)
python scripts/evaluate_tyre_model.py

# 4. Compare AI strategy to real team calls on 5 races
python scripts/strategy_report_card.py

# 5. Build a static playback file for one race
python scripts/build_static_race.py --year 2024 --gp Brazil --driver VER

# 6. Serve the static site locally
cd web
python -m http.server 8000
# Open http://localhost:8000
```

---

## Repo structure

```
f1-ai-race-engineer/
├── src/
│   ├── data/          # FastF1 loader, feature engineering
│   ├── models/        # XGBoost tyre model, Monte Carlo strategy
│   ├── commentary/    # Groq LLM persona + race-state formatter
│   └── replay/        # Lap-by-lap event detection
├── scripts/           # CLI utilities (train, evaluate, build)
├── web/               # Static site (deploys to GitHub Pages)
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── data/          # Pre-rendered race JSONs
├── models/            # Trained model + diagnostic plots
└── data/cache/        # FastF1 cache (gitignored)
```

---

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) — the unsung hero of F1 data analysis
- [Groq](https://groq.com) — for somehow giving away Llama 3.3 70B inference for free
- [David "Crofty" Croft](https://www.skysports.com/f1) — for the voice this project tries to imitate

---

## License

MIT. Built as a portfolio project — feel free to fork, learn from it, or use it as a starting point for your own analytics work.
