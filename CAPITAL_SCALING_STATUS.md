# Capital Scaling Status

Status: `GO`, authority: `NONE`, current level: `LEVEL 0`.

The repository now has a versioned capital ladder from shadow observation
through mature capital deployment. Promotion is always manual and cannot
exceed the evidence-based recommendation. Demotion is allowed automatically.

Current blockers:

- Phase 9 fill-and-close canary is not proven.
- No independent financial finalist exists.
- No real implementation-shortfall sample exists.

Implemented controls:

- account-equity and stop-distance sizing;
- whole-share rounding after cash, position, and liquidity caps;
- configurable drawdown throttle;
- ADV participation capacity;
- side-aware implementation shortfall;
- no margin, leverage, shorting, or withdrawals;
- atomic public artifacts and private level state;
- bounded retry for concurrent Windows artifact publication.

Canonical commands:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py capital status
.\.venv-ibkr\Scripts\python.exe .\main.py capital capacity
.\.venv-ibkr\Scripts\python.exe .\main.py capital recommend-level
.\.venv-ibkr\Scripts\python.exe .\main.py portfolio risk
.\.venv-ibkr\Scripts\python.exe .\main.py portfolio exposures
.\.venv-ibkr\Scripts\python.exe .\main.py portfolio capacity
.\.venv-ibkr\Scripts\python.exe .\main.py portfolio rebalance-preview
```
