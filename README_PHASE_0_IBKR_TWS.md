# IBKR Phase 0: installatie en read-only verbinding

## Doel

Deze fase doet maar vier dingen:

1. Een reproduceerbare Python 3.12-omgeving maken.
2. De officiële IBKR Native Python API installeren.
3. TWS alleen lokaal en in paper/read-only configureren.
4. Bewijzen dat Python met TWS kan verbinden zonder één financiële writecall.

Er wordt nog geen strategie, orderrouter, stop-loss of take-profit gebouwd. Dat zou nu te vroeg zijn.

## Bestanden

- `requirements.txt`: algemene runtime-, quant-, research- en testdependencies.
- `install_ibkr_windows.ps1`: maakt `.venv-ibkr`, installeert dependencies en zoekt de officiële `ibapi`.
- `.env.ibkr.example`: fail-closed paperconfiguratie.
- `ibkr_tws_probe.py`: read-only TCP/API-handshake. Bevat geen ordermethoden.
- `requirements.lock.txt`: wordt na installatie automatisch gegenereerd.

## Stap 1: Python

Installeer Python 3.12 x64. Controleer in PowerShell:

```powershell
py -0p
py -3.12 --version
```

Gebruik niet blind de nieuwste Python-major. Quantpackages en brokerlibraries lopen daar soms achteraan.

## Stap 2: officiële TWS API installeren

TWS als handelsprogramma is niet voldoende. Download en installeer ook de officiële TWS API voor Windows.

De Pythonbron staat daarna meestal hier:

```text
C:\TWS API\source\pythonclient
```

De installer installeert die map in dezelfde virtuele omgeving.

Gebruik bij voorkeur een TWS/API-combinatie met overeenkomstige versies. Voor API-ontwikkeling is de offline TWS-versie stabieler dan een automatisch bijgewerkte onlinevariant.

## Stap 3: bestanden in projectroot plaatsen

Plaats deze bestanden bijvoorbeeld in:

```text
C:\Users\alhar\Documents\datascraper
```

Open PowerShell in die map.

## Stap 4: installatie uitvoeren

Wanneer PowerShell scripts blokkeert, alleen voor de huidige sessie:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Daarna:

```powershell
.\install_ibkr_windows.ps1
```

Handmatig alternatief:

```powershell
py -3.12 -m venv .venv-ibkr
.\.venv-ibkr\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install "C:\TWS API\source\pythonclient"
python -m pip freeze > requirements.lock.txt
```

## Stap 5: TWS paperinstellingen

Log in op de paperomgeving van TWS.

Ga naar:

```text
Edit
→ Global Configuration
→ API
→ Settings
```

Zet in fase 0:

- `Enable ActiveX and Socket Clients`: aan.
- `Read-Only API`: aan laten.
- `Socket Port`: `7497`.
- `Allow connections from localhost only`: aan.
- API-logbestand: aan voor troubleshooting.
- Logging level: `Detail` tijdens ontwikkeling.

Gebruik nog niet poort `7496`. Dat is de standaard livepoort van TWS.

Laat order-precaution bypasses uit. Die zijn in deze fase niet nodig.

## Stap 6: env-bestand

De installer kopieert `.env.ibkr.example` naar `.env.ibkr`.

Controleer minimaal:

```dotenv
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=17
IBKR_READ_ONLY=true
IBKR_ORDER_AUTHORITY=NONE
IBKR_LIVE_TRADING_ENABLED=false
IBKR_ALLOW_ORDER_TRANSMISSION=false
```

Zet geen IBKR-gebruikersnaam of wachtwoord in `.env.ibkr`. TWS verzorgt de login.

## Stap 7: read-only probe

Zorg dat TWS open en ingelogd is. Voer uit:

```powershell
.\.venv-ibkr\Scripts\Activate.ps1
python .\ibkr_tws_probe.py --env-file .env.ibkr
```

Verwacht:

```text
"status": "IBKR_TWS_READ_ONLY_PROBE_GO"
```

Ook verwacht:

```text
"place_order": 0
"cancel_order": 0
"global_cancel": 0
```

De probe weigert:

- niet-lokale hosts;
- livepoorten;
- orderauthority anders dan `NONE`;
- ingeschakelde ordertransmissie.

## Veelvoorkomende fout 502

Dit betekent meestal dat de socketverbinding niet tot stand kwam.

Controleer:

1. TWS draait en is volledig ingelogd.
2. Je gebruikt de paperlogin.
3. `Enable ActiveX and Socket Clients` staat aan.
4. TWS en `.env.ibkr` gebruiken beide poort `7497`.
5. Alleen localhost is toegestaan.
6. Windows Firewall blokkeert Java/TWS of Python niet.
7. Geen ander proces gebruikt dezelfde client ID op dezelfde sessie.

## Marktdata

Voor verbinding en accountcallbacks is geen live koersabonnement nodig.

Voor koersen geldt later:

- `1`: live;
- `2`: frozen;
- `3`: delayed;
- `4`: delayed-frozen.

Delayed data kan voor technische ontwikkeling bruikbaar zijn, maar niet voor het bewijzen van executionkwaliteit of live alpha.

## ETF's en commodities in IBKR

IBKR classificeert gewone aandelen en de meeste ETF's als `STK`.

Commodityblootstelling kan bestaan uit:

- ETF/ETC als `STK`;
- futures als `FUT`.

Futures vereisen later contractdetails zoals multiplier, expiry, trading class, local symbol, minimum tick en first notice/last trade-logica. Een symbool als `GC` of `CL` is niet genoeg om veilig een contract te verhandelen.

## Bouwvolgorde na Phase 0

1. Read-only connection service in `main.py`.
2. Contract resolver voor `STK` en `FUT`.
3. Market calendar en session gate.
4. Historische datacollector met pacing en caching.
5. Point-in-time universe en corporate actions.
6. Backtest-engine met costs, slippage en benchmarks.
7. Risk engine en portfolio accounting.
8. Shadow signal runner.
9. IBKR paper order state machine.
10. Reconciliation, restart recovery en idempotency.
11. Supervised canary.
12. Pas daarna autonome authority.

TWS is goed voor ontwikkeling omdat je visueel kunt zien wat het programma doet. Voor een stabiele langdurige runtime kan later IB Gateway worden toegevoegd, maar niet voordat dezelfde adapter en veiligheidscontracten met TWS paper volledig zijn bewezen.
