# Web viewer

Static site that plays back a pre-rendered race. No server, no Python needed at runtime — everything ships in `docs/data/*.json`.

## Local preview

From the project root:

```powershell
# Build a race (requires .env with Groq key, takes ~2 min, ~25 LLM calls)
python scripts/build_static_race.py --year 2024 --gp Brazil --driver VER

# Serve the static site locally
cd docs
python -m http.server 8000
```

Open http://localhost:8000 in your browser.

## Deploy to GitHub Pages

1. Commit and push the repo to GitHub.
2. In repo Settings → Pages, set source to `main` branch and folder to `/docs`.
3. Your site goes live at `https://<your-username>.github.io/f1-ai-race-engineer/`.

## File layout

```
docs/
  index.html         # markup
  styles.css         # dark F1 theme
  app.js             # vanilla JS playback engine, Chart.js via CDN
  data/
    race_<year>_<gp>.json   # pre-rendered race
```
