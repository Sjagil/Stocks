# Local Operations UI

The Stocks operations console is a local, read-only view over the repository's
canonical public artifacts. It does not maintain a second ranking model and it
does not call IBKR.

## Start and stop

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py ui start `
  --host 127.0.0.1 `
  --port 8080

.\.venv-ibkr\Scripts\python.exe .\main.py ui status
.\.venv-ibkr\Scripts\python.exe .\main.py ui stop
```

Open `http://127.0.0.1:8080`.

Only loopback binding is accepted. The UI does not expose POST, PUT, PATCH, or
DELETE actions. HTTP responses use no-store, content-type, frame, referrer, and
content-security headers.

## Pages

- `/`: operations summary, broker observation, authority, risk, blockers.
- `/signals`: raw, diversified, stock, ETF, commodity, actionable, and
  auto-eligible collections plus cached-bar charts.
- `/universe`: point-in-time stocks, ETFs, commodity exposures, delisted
  records, eligibility, and pagination.
- `/sectors`, `/industries`, `/regions`: current opportunity rankings.
- `/etfs`, `/commodities`: explicit product and exposure classifications.
- `/strategies`: nested OOS research results and deployment blockers.
- `/portfolio`: current research allocation, rotation preview, caps, risk,
  correlation, and exposures.
- `/asset/{symbol}`: universe metadata, 1h/2h/4h/1d/1w/1mo indicator matrix,
  data freshness, cached charts, current portfolio context, and news links.
- `/news`: current multi-source market intelligence and official macro calendar
  links. News remains context-only.
- `/research`: experiment registry and bounded evidence summary.
- `/health`: runtime, IBKR, Phase 9, provider, and Telegram status.
- `/audit`: sanitized public audit artifacts.

The event stream at `/events` emits state fingerprints and cached dashboard
updates. Analysis coverage is cached for five minutes. Asset analysis reads
only the requested symbol's validated local bars.

## Asset analysis CLI

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py analysis coverage
.\.venv-ibkr\Scripts\python.exe .\main.py analysis asset --symbol AAPL
```

Every instrument in the universe returns its registered metadata. Technical
analysis is published only when validated local bars exist; otherwise the
result is explicitly `DATA_UNAVAILABLE`. No upsampling or synthetic intraday
bars are created.

## Authority

The console always reports the current canonical authority. At the time of this
contract:

```text
strategy_authority  NONE
execution_authority NONE
automatic_submit    false
```

An active UI or signals runtime is not active trading.

## Privacy

The viewmodel layer removes keys for raw account identifiers, account numbers,
credentials, passwords, API keys, secrets, tokens, fingerprint keys, and
private keys. It only reads files under `output/`, `runtime/heartbeat.json`, and
validated local bar caches. Private SQLite ledgers are not served.

## Validation

```powershell
$env:NODE_PATH = `
  "C:\Users\alhar\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"

& "C:\Users\alhar\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" `
  .\scripts\validate_ui.cjs
```

The script validates eleven desktop pages and five mobile layouts, loads real
cached charts, checks browser errors and overflow, and writes:

```text
output/ui/visual-validation.json
output/ui/screenshots/
output/ui/analytics/strategy-timeframe-sharpe.png
output/ui/analytics/strategy-timeframe-cost-pf.png
output/ui/analytics/portfolio-correlation.png
```

## Known data boundaries

- Stock issuer domicile is not inferred from listing venue.
- ETF issuer, expense ratio, AUM, holdings, and spread fields remain unavailable
  until a governed provider supplies them.
- Commodity technical watchlists are separately labelled
  `SIGNAL_CONFIDENCE_PROXY_NOT_PORTFOLIO_OPPORTUNITY_SCORE` when the portfolio
  opportunity layer has no commodity ranking.
- Commodity product structure and unresolved contracts block execution.
- Financial finalist status remains false until independent forward evidence
  and all frozen gates are satisfied.
