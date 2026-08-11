# IBKR Phase 1: read-only connection service

Phase 1 vervangt de losse probe als normale ingang. Gebruik vanaf nu:

```powershell
python .\main.py ibkr probe
python .\main.py ibkr status
python .\main.py ibkr watch --seconds 60
python .\main.py ibkr cycle --count 25
python .\main.py ibkr duplicate-client-check
python .\main.py ibkr disconnect-drill-preflight
python .\main.py ibkr disconnect-drill --seconds 180
python .\main.py doctor
```

De losse `ibkr_tws_probe.py` blijft alleen een diagnosetool voor Phase 0.

## Veiligheidscontract

Phase 1 is hard read-only:

```text
IBKR_READ_ONLY=true
IBKR_ORDER_AUTHORITY=NONE
IBKR_LIVE_TRADING_ENABLED=false
IBKR_ALLOW_ORDER_TRANSMISSION=false
IBKR_MAX_ORDER_NOTIONAL_EUR=0
IBKR_MAX_OPEN_ORDERS=0
IBKR_MAX_POSITIONS=0
```

Alleen lokale paperpoorten zijn toegestaan:

```text
TWS paper        7497
IB Gateway paper 4002
```

Livepoorten `7496` en `4001` worden door configuratievalidatie geblokkeerd.
`IBKR_HOST` wordt voor Phase 1 gecanonicaliseerd naar exact `127.0.0.1`; `localhost` is toegestaan als invoer, maar IPv6 `::1`, whitespace en externe hosts worden geweigerd zodat preflight en freeze-verifier dezelfde host-evidence gebruiken.
`IBKR_ORDER_AUTHORITY` wordt gecanonicaliseerd naar exact `NONE`; lowercase `none` is toegestaan als invoer, maar whitespace of iedere andere authority blokkeert de configuratie voordat preflight of drill start.

Booleanwaarden in `.env.ibkr` zijn strikt: gebruik expliciet `true` of `false`-achtige tokens. Typo's zoals `maybe`, `disabled` of `tru` worden geweigerd voordat preflight of drill start.
`APP_ENV` is in Phase 1 beperkt tot `development`, `test` of `paper`; `production`, `live`, onbekende labels en omliggende whitespace worden geweigerd.
Security-type- en currency-allowlists zijn ook fail-closed: Phase 1/2 accepteert alleen `STK,FUT` en `EUR,USD`; lege lijsten of scope-uitbreidingen zoals `OPT` of `JPY` blokkeren de configuratie.
Numerieke `.env.ibkr`-waarden accepteren alleen kale integer- of decimale tokens zonder omliggende whitespace, booleanwoorden, `nan`, exponentnotatie of lijstachtige substituten. `IBKR_MARKET_DATA_TYPE=3` is verplicht in Phase 1; live, frozen en delayed-frozen types worden geblokkeerd totdat een latere datafase die authority expliciet opent.
Heartbeat-, stale- en reconnectinstellingen worden uit `.env.ibkr` geladen en fail-closed gevalideerd: timingwaarden moeten eindig en positief zijn, `IBKR_STALE_AFTER_SECONDS` moet groter zijn dan de heartbeatinterval, `IBKR_MAX_RECONNECT_ATTEMPTS` moet positief zijn en de reconnect-delaylijst mag niet leeg zijn of nul/oneindige waarden bevatten.
`IBKR_OUTPUT_DIR=output/ibkr` is verplicht wanneer de configuratie uit `.env.ibkr` wordt geladen. Absolute paden, parent traversal en alternatieve artifactroots worden geweigerd, zodat Phase 1-bewijs en freeze-artifacts onder dezelfde projectrootlocatie blijven.
`IBKR_ACCOUNT` moet leeg blijven in Phase 1. Accountidentiteit komt alleen via read-only TWS-callbacks binnen en wordt uitsluitend als `managed_account_count` gemaskeerd gerapporteerd.
`IBKR_LOG_LEVEL` accepteert alleen `CRITICAL`, `ERROR`, `WARNING`, `INFO` of `DEBUG`; lowercase wordt gecanoniseerd en onbekende waarden of omliggende whitespace blokkeren de configuratie.

## Healthstatus

```text
HEALTHY
  socket actief, eventthread actief, handshake voltooid,
  serverklok actueel en geen fatale fouten

DEGRADED
  centrale API-sessie actief, maar een datafarm meldde tijdelijk verlies

STALE
  geen geldige heartbeat binnen de stale-threshold

DISCONNECTED
  socket of eventthread is niet actief

FAILED
  fatale fout of maximum reconnectpogingen overschreden
```

Heartbeat gebruikt `reqCurrentTime` en draait standaard iedere 25 seconden. Een heartbeat ouder dan 45 seconden is stale.
`python .\main.py ibkr status` doet een single-attempt health snapshot. Wanneer TWS paper niet luistert op `127.0.0.1:7497`, rapporteert het command fail-fast `FAILED` met IBKR-fout 502 en nul financiële calls in plaats van de volledige reconnect-delayreeks af te wachten.

## Reconnectbeleid

```text
1e poging  na 2 seconden
2e poging  na 5 seconden
3e poging  na 15 seconden
4e+ poging na 30 seconden
maximum    5 pogingen
daarna     fail closed
```

Er is geen eindeloze reconnectloop.

## Statusartifacts

De service schrijft append-only JSONL-statusregels naar:

```text
output/ibkr/ibkr-status-YYYYMMDD.jsonl
```

Accountnummers worden niet opgeslagen. Alleen `managed_account_count` wordt gerapporteerd.
`python .\main.py ibkr disconnect-drill ...` schrijft daarnaast zelf een machine-readable drillartifact naar:

```text
output/ibkr/phase1-disconnect-drill-YYYYMMDD-HHMMSS.json
```

Ook `NO_GO`-drills worden vastgelegd. Een artifact is pas freezewaardig wanneer het werkelijk `disconnect_detected=true`, `disconnected_or_stale_state_seen=true`, `bounded_reconnect_attempted=true`, `reconnect_recovered=true`, `thread_leak=false` en financiële counters op nul bevat.

## Lokale verificatie op 2026-07-20

Uitgevoerd via `.venv-ibkr`:

```text
python .\main.py doctor                         GO
python .\main.py ibkr probe                     HEALTHY
python .\main.py ibkr status                    HEALTHY
python .\main.py ibkr watch --seconds 60        HEALTHY
python .\main.py ibkr cycle --count 25          GO
python .\main.py ibkr duplicate-client-check    GO
python .\main.py ibkr disconnect-drill-preflight --skip-socket-check
                                                   GO
python .\main.py ibkr disconnect-drill --seconds 8 --poll-seconds 2
                                                   NO_GO, no disconnect observed
python .\main.py ibkr disconnect-drill --seconds 8 --poll-seconds 2
                                                   NO_GO, no disconnect observed,
                                                   artifact written by main.py
.\scripts\run_phase1_disconnect_drill.ps1 -Seconds 180 -PollSeconds 2
                                                   NO_GO, no disconnect observed
python -m pytest -q                             355 passed
python -m ruff check .\main.py .\src .\tests    GO
forbidden write-method scan                     GO
account identifier scan                         GO
live IBKR data request scan                     GO
```

Bewezen runtimewaarden:

```text
server_version          225
managed_account_count   1
errors                  0
financial_calls         all 0
thread_leak             false
connect_disconnect_25   GO
duplicate_client_id     error 326 detected
no_event_drill          NO_GO without false positive
canonical_freeze_drill  output/ibkr/phase1-disconnect-drill-20260720-133116.json
phase1_freeze_report    PHASE1_FREEZE_REPORT.md
```

## Frozen GO

De echte TWS paper disconnect/reconnectdrill is bewezen en door de applicatiegate geaccepteerd:

```text
IBKR_PHASE1_CONNECTION_SERVICE_FROZEN_GO
```

Zie ook:

```text
docs/PHASE1_CONNECTION_SERVICE_FROZEN.md
```

De applicatiegate leest niet alleen de marker, maar controleert ook het artifactpad, de artifactnaam en timestamp, de SHA256, TWS paper `127.0.0.1:7497`, de JSON-inhoud, de statusvolgorde en de huidige hashes van de bevroren Phase 0/1-bestanden voordat Phase 2 read-only resolvergedrag wordt vrijgegeven. De bevroren Phase 1-set omvat `main.py`, application-config/context/lifecycle/gatecode en de IBKR connection/client/callback/error/health modules.

Static audit voor alle niet-externe gates:

```powershell
.\scripts\run_phase1_static_audit.ps1
```

Deze schrijft `output/ibkr/phase1-static-audit-YYYYMMDD-HHMMSS.json`. Dit vervangt de handmatige TWS-disconnectdrill niet. De audit valideert ook dat de publieke `.env.ibkr.example`-bestanden door dezelfde fail-closed configparser komen.
Als `PHASE1_FREEZE_REPORT.md` bestaat, valideert de static audit dat rapport via dezelfde applicatiegate die Phase 2 live resolvergedrag vrijgeeft. Een aanwezig maar ongeldig freeze-rapport maakt de audit `NO_GO`.

De freeze-gate rapporteert nu `PHASE1_FROZEN`; Phase 2 mag de centrale service importeren voor read-only contractidentiteit.

Offline Phase 3-voorbereiding leest alleen lokale contractcache-uren. De markturenrapporten tonen bekende cachedekking, expliciete `CLOSED`-dagen, futuresachtige overnight windows, `NO_KNOWN_NEXT_OPEN` wanneer de cache geen toekomstig handelsvenster meer bevat, en een lokale `exchange-calendars` cross-check met `MATCH`, `MISMATCH` of `UNSUPPORTED_CALENDAR`.

Offline Phase 4-voorbereiding is beperkt tot bar-schema en statusrapportage:

```powershell
python .\main.py data bars schema
python .\main.py data bars status
python .\main.py data bars init-cache
python .\main.py data bars validate-cache
python .\main.py data bars request-policy
```

Deze commands doen geen IBKR- of EODHD-calls. EODHD blijft uitgeschakeld tot een latere datafase expliciet read-only data-authority geeft. EODHD-planning is voorlopig daily en STK-only; futuresbars blijven IBKR-only totdat futures-chain-, roll- en provider-symbolregels zijn gedefinieerd. `init-cache` schrijft alleen `data/bars/bar_manifest.json`; de lokale cachewriter weigert shards met out-of-order timestamps, duplicate timestamps of exacte intraday-gaps. `validate-cache` controleert bestaande lokale `bars.parquet` partities, strikte enumwaarden voor `sec_type`, `interval`, `data_type` en `source`, runtime enumwaarden in requests, bars en cachepaden, timezone-aware runtime `datetime` waarden in requests en bars, runtime `Decimal` OHLC-waarden en integer/null volume, positieve integer-`con_id` waarden in requestplanning, cachepad, partitiepad en record, parseerbare ISO-8601 timestampvelden zonder booleans of omliggende whitespace, strikte OHLC-decimals zonder booleans of omliggende whitespace, manifestdrift, point-in-time `available_at` ordering en, zodra barfiles bestaan, of iedere `(sec_type, con_id)` in de lokale Phase 2-contractcache bestaat. De request-planner kan dezelfde contractidentiteitscontrole toepassen wanneer lokale contractrows worden meegegeven. `request-policy` rapporteert alleen strikt integer-getypeerde planningregels voor concurrency, retry, timeout en deduplicatie; execution blijft uit.
