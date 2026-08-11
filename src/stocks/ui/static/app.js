document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) window.lucide.createIcons();

  const refresh = document.getElementById("refresh-state");
  const events = new EventSource("/events");
  events.addEventListener("state", (event) => {
    const payload = JSON.parse(event.data);
    if (refresh) {
      const mode = payload.dashboard?.runtime?.mode || "UNKNOWN";
      refresh.textContent = `State updated ${new Date(payload.timestamp).toLocaleTimeString()} · ${mode}`;
    }
  });
  events.onerror = () => {
    if (refresh) refresh.textContent = "State stream reconnecting";
  };

  let symbol = null;
  let interval = "1d";
  const chart = document.getElementById("price-chart");
  if (chart?.dataset.initialSymbol) {
    symbol = chart.dataset.initialSymbol;
  }
  document.querySelectorAll("[data-chart-symbol]").forEach((button) => {
    button.addEventListener("click", () => {
      symbol = button.dataset.chartSymbol;
      loadChart();
    });
  });
  document.querySelectorAll("[data-interval]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-interval]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      interval = button.dataset.interval;
      if (symbol) loadChart();
    });
  });

  async function loadChart() {
    if (!chart || !symbol || !window.Plotly) return;
    chart.textContent = "Loading cached bars";
    const response = await fetch(`/api/chart/${encodeURIComponent(symbol)}?interval=${interval}&limit=300`);
    const data = await response.json();
    if (data.status !== "GO" || !data.bars.length) {
      chart.textContent = `No validated ${interval} bars for ${symbol}`;
      return;
    }
    chart.textContent = "";
    const x = data.bars.map((bar) => bar.timestamp);
    window.Plotly.react(chart, [
      {
        type: "candlestick",
        x,
        open: data.bars.map((bar) => bar.open),
        high: data.bars.map((bar) => bar.high),
        low: data.bars.map((bar) => bar.low),
        close: data.bars.map((bar) => bar.close),
        increasing: {line: {color: "#72c98b"}},
        decreasing: {line: {color: "#e26d6d"}},
        name: symbol
      }
    ], {
      margin: {l: 48, r: 18, t: 18, b: 35},
      paper_bgcolor: "#171c1f",
      plot_bgcolor: "#171c1f",
      font: {color: "#94a1a6", size: 11},
      xaxis: {rangeslider: {visible: false}, gridcolor: "#252d30"},
      yaxis: {gridcolor: "#252d30", fixedrange: false},
      showlegend: false
    }, {responsive: true, displaylogo: false});
    const title = document.getElementById("chart-title");
    if (title) title.textContent = `${symbol} · ${interval} · ${data.provider} · ${data.bar_origin}`;
  }
  if (symbol && chart) loadChart();

  const exchangeMap = document.getElementById("exchange-map");
  if (exchangeMap && window.Plotly) {
    renderExchangeGlobe(JSON.parse(exchangeMap.dataset.exchanges || "[]"));
  }

  function renderExchangeGlobe(exchanges) {
    if (!exchangeMap || !window.Plotly) return;
    window.Plotly.react(exchangeMap, [{
      type: "scattergeo",
      mode: "markers",
      lon: exchanges.map((row) => row.longitude),
      lat: exchanges.map((row) => row.latitude),
      text: exchanges.map((row) => row.name),
      customdata: exchanges.map((row) => [
        row.status,
        row.local_time,
        row.next_event_type || "",
        row.next_event_local || ""
      ]),
      hovertemplate: "%{text}<br>%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} %{customdata[3]}<extra></extra>",
      marker: {
        size: exchanges.map((row) => row.status === "OPEN" ? 11 : 8),
        color: exchanges.map((row) => row.status === "OPEN" ? "#72c98b" : "#e4b45d"),
        line: {color: "#101416", width: 1.5}
      }
    }], {
      margin: {l: 0, r: 0, t: 0, b: 0},
      paper_bgcolor: "#171c1f",
      font: {color: "#b9c6ca", size: 9},
      showlegend: false,
      geo: {
        projection: {type: "orthographic", rotation: {lon: 15, lat: 12}},
        showland: true,
        landcolor: "#263033",
        showocean: true,
        oceancolor: "#101719",
        showlakes: true,
        lakecolor: "#101719",
        showcountries: true,
        countrycolor: "#3a474b",
        coastlinecolor: "#526065",
        bgcolor: "#171c1f"
      }
    }, {
      responsive: true,
      displaylogo: false,
      topojsonURL: "/static/"
    });
  }

  function updateExchangeState(exchanges) {
    const byName = new Map(exchanges.map((row) => [row.name, row]));
    document.querySelectorAll(".exchange-card[data-exchange]").forEach((card) => {
      const row = byName.get(card.dataset.exchange);
      if (!row) return;
      card.classList.remove("open", "closed", "calendar_unavailable");
      card.classList.add(String(row.status || "CALENDAR_UNAVAILABLE").toLowerCase());
      const status = card.querySelector(".exchange-status");
      if (status) status.textContent = row.status;
      const nextEvent = card.querySelector(".exchange-next-event");
      if (nextEvent) {
        nextEvent.textContent = `${row.country} · ${row.next_event_type || "—"} ${row.next_event_local || "—"}`;
      }
    });
    const summary = document.getElementById("exchange-status-summary");
    if (summary) {
      const open = exchanges.filter((row) => row.status === "OPEN").length;
      const closed = exchanges.filter((row) => row.status === "CLOSED").length;
      summary.textContent = `${open} open · ${closed} closed`;
      summary.classList.toggle("actionable", open > 0);
      summary.classList.toggle("persistent", open === 0);
    }
    renderExchangeGlobe(exchanges);
  }

  async function refreshExchangeState() {
    if (!exchangeMap) return;
    try {
      const response = await fetch("/api/dimensions/region", {cache: "no-store"});
      if (!response.ok) return;
      const payload = await response.json();
      if (Array.isArray(payload.exchange_clock)) {
        updateExchangeState(payload.exchange_clock);
      }
    } catch (_error) {
      const summary = document.getElementById("exchange-status-summary");
      if (summary) summary.textContent = "Calendar refresh delayed";
    }
  }

  function updateExchangeClocks() {
    document.querySelectorAll("[data-timezone]").forEach((card) => {
      const target = card.querySelector(".exchange-local-time");
      if (!target) return;
      target.textContent = new Intl.DateTimeFormat(undefined, {
        timeZone: card.dataset.timezone,
        weekday: "short", hour: "2-digit", minute: "2-digit", second: "2-digit"
      }).format(new Date());
    });
  }
  updateExchangeClocks();
  if (document.querySelector("[data-timezone]")) setInterval(updateExchangeClocks, 1000);
  if (exchangeMap) setInterval(refreshExchangeState, 60000);
});
