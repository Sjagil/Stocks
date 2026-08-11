# IBKR Phase 2 Contract Resolver

Phase 2 resolves IBKR contract identity only. It is broker read-only and depends on the frozen Phase 1 connection service.

## CLI

```powershell
python .\main.py ibkr contract status
python .\main.py ibkr contract schema
python .\main.py ibkr contract init-cache
python .\main.py ibkr contract validate-cache
python .\main.py ibkr contract resolve-stock --symbol AAPL --currency USD --exchange SMART --primary-exchange NASDAQ
python .\main.py ibkr contract resolve-stock --symbol ASML --currency EUR --exchange SMART --primary-exchange AEB
python .\main.py ibkr contract resolve-stock --symbol SPY --asset-class etf --currency USD --exchange SMART --primary-exchange ARCA
python .\main.py ibkr contract resolve-future --symbol GC --currency USD --exchange COMEX
```

## Resolution Rules

```text
0 matches                      NOT_FOUND
1 match                        RESOLVED
more than 1                    AMBIGUOUS_BLOCKED
```

The resolver never selects the first IBKR result automatically. Ambiguous futures chains are written to the error audit and are not cached as resolved contracts.

## Storage

```text
output/ibkr/contracts/stocks.parquet
output/ibkr/contracts/futures.parquet
output/ibkr/contracts/contract_requests.jsonl
output/ibkr/contracts/contract_errors.jsonl
output/ibkr/contracts/contract_manifest.json
```

Required STK identity:

```text
conId
symbol
localSymbol
secType
currency
exchange
primaryExchange
tradingClass
minTick
validExchanges
timeZoneId
tradingHours
liquidHours
marketRuleIds
longName
```

Required FUT identity:

```text
conId
symbol
localSymbol
secType
currency
exchange
lastTradeDateOrContractMonth
realExpirationDate
lastTradeTime
multiplier
minTick
tradingClass
underConId
timeZoneId
tradingHours
liquidHours
marketRuleIds
```

## Safety

Allowed calls:

```text
reqContractDetails
reqMatchingSymbols
reqMarketRule
```

Current live validation used only `reqContractDetails`.

Forbidden calls:

```text
placeOrder
cancelOrder
reqGlobalCancel
reqMktData
reqHistoricalData
reqRealTimeBars
```

No account identifiers, credentials, order objects, market data or historical bars are written by Phase 2.
