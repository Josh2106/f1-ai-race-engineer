// F1 AI Race Engineer — static playback engine.
//
// Loads a pre-rendered race JSON and animates it lap-by-lap.

const $ = (sel) => document.querySelector(sel);

let raceData = null;
let timer = null;
let currentLap = 0;
let chart = null;

const TEAM_COLORS = {
  "Red Bull Racing": "#1E5BC6",
  "Mercedes":        "#00D2BE",
  "Ferrari":         "#DC0000",
  "McLaren":         "#FF8700",
  "Aston Martin":    "#006F62",
  "Alpine":          "#0090FF",
  "Williams":        "#005AFF",
  "RB":              "#6692FF",
  "Kick Sauber":     "#52E252",
  "Haas F1 Team":    "#B6BABD",
  "AlphaTauri":      "#2B4562",
  "Alfa Romeo":      "#900000",
};
const colorFor = (team) => TEAM_COLORS[team] || "#888";

async function loadRace(path) {
  $("#race-info").textContent = "Loading race data...";
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`Failed to load ${path}`);
  raceData = await resp.json();

  const m = raceData.meta;
  $("#race-info").innerHTML =
    `${m.year} ${m.gp} GP &middot; Focus: <b>${m.focus_driver}</b> (${m.focus_team}) &middot; started on ${m.starting_compound}`;
  $("#total-laps").textContent = m.total_laps;

  renderStrategy(raceData.strategy);
  resetPlayback();
  initChart();
}

function renderStrategy(strats) {
  const html = strats.map((s, i) => `
    <div class="strat ${i === 0 ? 'best' : ''}">
      <span><b>#${s.rank}</b> &middot; ${s.summary.replace(/total=[\d.]+s/, '')}</span>
      <span class="total">${s.total_s.toFixed(0)}s</span>
    </div>`).join("");
  $("#strategy-list").innerHTML = html;
}

function renderStandings(standings) {
  const rows = standings.map(s => {
    const gap = s.pos === 1 ? "LEADER" : `+${s.gap.toFixed(1)}s`;
    const cls = s.pos === 1 ? "leader" : "";
    const color = colorFor(s.team);
    return `<tr class="${cls}">
      <td class="pos">P${s.pos}</td>
      <td><span style="color:${color}; font-weight:600;">${s.driver}</span></td>
      <td>${s.team}</td>
      <td class="gap">${gap}</td>
    </tr>`;
  }).join("");
  $("#standings tbody").innerHTML = rows;
}

function appendCommentary(lap, event) {
  const feed = $("#commentary-feed");
  const div = document.createElement("div");
  div.className = "msg";
  div.innerHTML = `
    <span class="meta">Lap ${lap} &middot; ${event.type.replace("_", " ")} &middot; ${event.driver || ""}</span>
    ${event.text}
  `;
  feed.prepend(div);  // newest on top
}

function initChart() {
  const ctx = document.getElementById("lap-chart").getContext("2d");
  if (chart) chart.destroy();

  // Pick top-5 drivers (by final standings of last lap)
  const lastLap = raceData.laps[raceData.laps.length - 1];
  const topDrivers = lastLap.standings.slice(0, 5);

  const datasets = topDrivers.map(d => ({
    label: d.driver,
    borderColor: colorFor(d.team),
    backgroundColor: colorFor(d.team) + "33",
    data: [],
    tension: 0.3,
    pointRadius: 0,
    borderWidth: 2,
  }));

  chart = new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#8892b0", font: { size: 11 } } },
      },
      scales: {
        x: {
          title: { display: true, text: "Lap", color: "#8892b0" },
          ticks: { color: "#8892b0" },
          grid: { color: "#232a3d" },
        },
        y: {
          title: { display: true, text: "Gap to leader (s)", color: "#8892b0" },
          ticks: { color: "#8892b0" },
          grid: { color: "#232a3d" },
        },
      },
    },
  });
}

function pushChartLap(lap, standings) {
  if (!chart) return;
  chart.data.labels.push(lap);
  chart.data.datasets.forEach(ds => {
    const driverRow = standings.find(s => s.driver === ds.label);
    ds.data.push(driverRow ? driverRow.gap : null);
  });
  chart.update("none");
}

function tickLap() {
  if (!raceData || currentLap >= raceData.laps.length) {
    stop();
    $("#btn-play").textContent = "✓ Finished";
    return;
  }

  const lap = raceData.laps[currentLap];
  $("#lap-num").textContent = lap.lap;
  renderStandings(lap.standings);
  pushChartLap(lap.lap, lap.standings);
  lap.events.forEach(ev => appendCommentary(lap.lap, ev));
  currentLap++;
}

function play() {
  if (timer) return;
  const interval = parseInt($("#speed").value, 10);
  timer = setInterval(tickLap, interval);
  $("#btn-play").textContent = "⏸ Pause";
}

function pause() {
  clearInterval(timer);
  timer = null;
  $("#btn-play").textContent = "▶ Play";
}

function stop() { pause(); }

function resetPlayback() {
  pause();
  currentLap = 0;
  $("#lap-num").textContent = "--";
  $("#standings tbody").innerHTML = "";
  $("#commentary-feed").innerHTML = "";
  $("#btn-play").textContent = "▶ Play";
  if (chart) {
    chart.data.labels = [];
    chart.data.datasets.forEach(d => d.data = []);
    chart.update("none");
  }
}

// Wire up
$("#btn-play").addEventListener("click", () => (timer ? pause() : play()));
$("#btn-restart").addEventListener("click", () => { resetPlayback(); });
$("#speed").addEventListener("change", () => {
  if (timer) { pause(); play(); }
});
$("#race-pick").addEventListener("change", (e) => loadRace(e.target.value));

// Initial load
loadRace($("#race-pick").value).catch(err => {
  $("#race-info").textContent = `Error: ${err.message}`;
});
