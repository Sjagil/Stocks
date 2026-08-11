# Signals and Autopilot Status

```text
SIGNALS_CAN_RUN_WITHOUT_BROKER         true
SIGNALS_INCLUDE_STOP_LOSS              true
SIGNALS_INCLUDE_TAKE_PROFIT            true
MANUAL_EXECUTION_SUPPORTED             true
AUTOPILOT_CONTINUOUS_RESEARCH          true
AUTOPILOT_AUTO_LIVE_PROMOTION          false
SIGNAL_AUTHORITY_SEPARATE_FROM_EXECUTION true
```

Current evidence:

```text
historical survivors                   16
research candidates                     6
frozen shadow                          10
manual actionable                       0
paper candidates                        0
live-canary candidates                  0
followed local-cache assets            201
shadow signal plans                     25
broker calls                             0
orders generated                         0
```

The first bounded autopilot cycle completed without runtime failures. Its
canonical point-in-time campaign was `DATA_BLOCKED` because historical
Shariah eligibility is unavailable. Survivor recovery and shadow signal
generation remain operational, but no strategy received execution authority.

Phase 9 submit/cancel evidence is available. Phase 9 fill/close remains an
operator-run blocker. Live preflight and the persistent kill switch are
implemented; the live writer is not frozen and cannot send an order.
