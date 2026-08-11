# IBKR Phase 0 Repair v2

## Doel

Deze reparatie doet twee dingen:

1. De oude of verkeerde IBKR-gerelateerde Pythonpackages uit `.venv-ibkr` verwijderen.
2. De officiële lokale IBKR Python API installeren vanuit `C:\TWS API\source\pythonclient`.

De probe is vervangen door `ibkr_tws_probe.py` met schema `ibkr_tws_read_only_probe_v2`. Deze versie converteert `bytes` recursief naar tekst en gebruikt `default=str` als laatste vangnet bij JSON-output.

## Officiële API-bron

Gebruik op dit moment `TWS API Latest for Windows`. De officiële downloadpagina vermeldt `API 10.48`, release date `Jul 7 2026`, en geeft aan dat de Windows Latest-package de Python API bevat. De Stable Windows-package `10.45` bevat volgens dezelfde pagina geen Python API.

Bronnen:

- https://interactivebrokers.github.io/
- https://ibkrcampus.com/campus/trading-lessons/accessing-the-tws-python-api-source-code/
- https://interactivebrokers.github.io/tws-api/connection.html

## Niet meer installeren

Gebruik voor deze repository niet:

```powershell
pip install ib
pip install iba
pip install ibapi
pip install ib_async
pip install ib_insync
```

`ibapi` moet uitsluitend uit de officiële lokale TWS API-bron komen:

```text
C:\TWS API\source\pythonclient
```

## Uitvoeren

Controleer eerst:

```powershell
Test-Path "C:\TWS API\source\pythonclient"
Get-ChildItem "C:\TWS API\source\pythonclient"
```

Daarna:

```powershell
cd C:\Users\alhar\Documents\Stocks
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\repair_ibkr_phase0_v2.ps1
```

## Herkomst controleren

```powershell
.\.venv-ibkr\Scripts\Activate.ps1
python -c "import ibapi; print(ibapi.__file__)"
python -m pip show ibapi
python -m pip check
```

Het importpad moet naar deze project-venv wijzen:

```text
C:\Users\alhar\Documents\Stocks\.venv-ibkr\Lib\site-packages\ibapi\__init__.py
```

## TWS socket controleren

TWS moet met paperaccount draaien.

Instellingen:

```text
Enable ActiveX and Socket Clients      aan
Read-Only API                          aan
Socket Port                            7497
Allow connections from localhost only aan
```

Controleer de lokale poort:

```powershell
Get-NetTCPConnection -LocalPort 7497 -State Listen
Test-NetConnection 127.0.0.1 -Port 7497
```

Als `TcpTestSucceeded` niet `True` is, luistert TWS niet correct en hoeft Python nog niet gedebugd te worden.

## Probe uitvoeren

```powershell
cd C:\Users\alhar\Documents\Stocks
.\.venv-ibkr\Scripts\Activate.ps1
python .\ibkr_tws_probe.py --env-file .env.ibkr
```

Geslaagd betekent minimaal:

```json
{
  "schema": "ibkr_tws_read_only_probe_v2",
  "status": "IBKR_TWS_READ_ONLY_PROBE_GO",
  "api_ready": true,
  "financial_calls": {
    "place_order": 0,
    "cancel_order": 0,
    "global_cancel": 0
  }
}
```

De probe blokkeert nog steeds livepoorten, niet-lokale hosts, orderauthority anders dan `NONE` en ordertransmissie.
