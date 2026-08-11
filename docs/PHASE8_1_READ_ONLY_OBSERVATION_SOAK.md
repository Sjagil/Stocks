# Phase 8.1 Read-Only Observation Soak

Phase 8.1 hardens broker observation by repeating the frozen Phase 8 snapshot flow. It measures snapshot completeness, non-atomic state changes, callback integrity, subscription cleanup, privacy, and operator-assisted reconnect recovery.

Storage separation:

```text
public artifacts: output/ibkr/phase8_1/
private store:    data/broker/phase8_1/private/observation_soak.sqlite3
```

Ledger roles remain separate:

```text
PHASE7_SYNTHETIC_LEDGER
PHASE8_BROKER_OBSERVATION_STORE
PHASE8_1_BROKER_BASELINE
```

The Phase 8.1 baseline is an observation only. It grants no ownership over broker state and never mutates the Phase 7 synthetic ledger.
