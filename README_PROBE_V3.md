# IBKR read-only probe v3

## Oorzaak

IBKR API 10.48 gebruikt voor serverfouten een nieuwere callbackvorm:

```text
error(reqId, errorTime, errorCode, errorMsg, advancedOrderRejectJson)
```

De probe v2 implementeerde de oudere vorm zonder `errorTime`. Daardoor kreeg de callback één argument te veel en stopte de IBKR eventthread.

De socketverbinding en handshake werkten al:

- `api_ready=true`;
- `server_version=225`;
- geldige `connection_time`;
- officiële `ibapi 10.48.1`.

## Installeren

Pak de bestanden uit in dezelfde tijdelijke map en voer uit:

```powershell
cd <map-met-deze-bestanden>
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\apply_probe_v3.ps1
```

Het patchscript:

- maakt een backup van probe v2;
- vervangt de actieve probe;
- draait een compilecheck;
- scant op echte ordermethodes;
- start daarna automatisch de probe.

## Handmatig

Kopieer `ibkr_tws_probe.py` naar:

```text
C:\Users\alhar\Documents\Stocks\ibkr_tws_probe.py
```

Voer daarna uit:

```powershell
cd C:\Users\alhar\Documents\Stocks
.\.venv-ibkr\Scripts\Activate.ps1
python .\ibkr_tws_probe.py --env-file .env.ibkr
```

Verwacht:

```text
"schema": "ibkr_tws_read_only_probe_v3"
"status": "IBKR_TWS_READ_ONLY_PROBE_GO"
"api_ready": true
"event_loop_alive": true
"managed_account_count": minimaal 1
"place_order": 0
"cancel_order": 0
"global_cancel": 0
```
