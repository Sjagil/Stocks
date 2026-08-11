# Operational Machine Status

## Canonical runtime

`main.py run` is the single continuous runtime entrypoint. The current-user
Windows task `Stocks Canonical Runtime` invokes it in `SIGNALS_ONLY` mode.
Heartbeat state is published atomically at `runtime/heartbeat.json`.

The previous `Stocks Bounded Autopilot` scheduled task is disabled to prevent
duplicate supervisors. Execution authority remains `NONE`; no paper or live
writer is enabled by the runtime task.

```text
DYNAMIC_MULTI_STRATEGY_ENGINE       GO
BOUNDED_OPERATIONAL_SUPERVISOR      GO
WINDOWS_SCHEDULED_TASK              GO
LAST_SCHEDULED_TASK_RESULT          0
LAST_SCHEDULED_CYCLE_SECONDS        7.14
SCHEDULER_PRIORITY                  4_NORMAL
SCHEDULER_STDIO_REDIRECT            GO
COMPONENT_TIMEOUT_PROPAGATION       GO
PROCESS_TREE_TIMEOUT_CLEANUP        GO
PUBLIC_HEARTBEAT_CURRENT            true
PHASE1_FREEZE_INTEGRITY             GO
PAPER_AUTOTRADE_IMPLEMENTED         PARTIAL_CONTROL_PLANE
PAPER_AUTOTRADE_ENABLED             false
PAPER_FILL_CLOSE_CANARY_GO          false
PAPER_RECONCILIATION_GO             true
LIVE_CANARY_IMPLEMENTED             PREFLIGHT_ONLY
LIVE_CANARY_READY                   false
LIVE_CANARY_ENABLED                 false
LIVE_AUTOSCALE_ENABLED              false
RESEARCH_AUTOPILOT_ENABLED          true
RESEARCH_AUTOPILOT_AUTO_LIVE_PROMOTION false
BACKTEST_POSITIVE_PAIRS             62
BACKTEST_POSITIVE_UNIQUE_STRATEGIES 28
STRICT_RESEARCH_SHORTLIST           10
ECONOMICALLY_INDEPENDENT_SHORTLIST  8
FINANCIAL_FINALIST_GO               false
CAPITAL_SCALING_ENGINE              true
CURRENT_CAPITAL_LEVEL               0
REAL_LIVE_ORDER_PLACED              false
```

The machine can run data, signals, macro status, portfolio construction,
reconciliation, research and notifications as one bounded cycle. Automatic
paper submission remains blocked by the outstanding real fill-and-close
canary. Live submission remains blocked by the unfrozen live writer and live
operator gates.

Heavy steps are due-gated: daily at most once per 20 hours, dynamic at most
once per hour, and market data plus macro refreshes at most once per six
hours. A component timeout degrades the cycle, does not advance its refresh
clock, terminates its subprocess tree, and is published in both the cycle and
machine status. `cycle-progress.json` provides an atomic step heartbeat.
