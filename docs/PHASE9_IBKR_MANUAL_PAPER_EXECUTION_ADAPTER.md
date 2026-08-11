# Phase 9 IBKR Manual Paper Execution Adapter

Phase 9 is a manual-only TWS Paper canary adapter. It connects the frozen Phase 7 execution-control-plane concepts to a real IBKR paper writer path, but it does not authorize any strategy, shadow runner, or live account.

## Authority

- Active execution authority remains `NONE` until all Phase 9 evidence is complete.
- Target authority after full evidence is `MANUAL_PAPER_CANARY`.
- Strategy authority, shadow authority, and live authority remain `NONE`.
- `FINANCIAL_FINALIST_GO` remains `false`.

## Allowed Broker Write Surface

- `placeOrder` may only be called from `src/stocks/ibkr/paper_execution/submission.py`.
- `cancelOrder` may only be called from `src/stocks/ibkr/paper_execution/cancellation.py`.
- `reqIds` may only be called from `src/stocks/ibkr/paper_execution/order_ids.py`.
- `reqGlobalCancel`, `reqAutoOpenOrders`, `exerciseOptions`, market data, realtime bars, and historical data are blocked for Phase 9.

## Manual Canary Flow

1. `phase9 preflight` must be `GO`.
2. `phase9 prepare` creates one manual operator intent.
3. `phase9 approve` records one exact approval challenge.
4. `phase9 submit` may make one paper `placeOrder` call after all gates pass.
5. `phase9 prepare-cancel` and `phase9 approve-cancel` are separate from submit approval.
6. `phase9 cancel` may make one same-client paper `cancelOrder` call.
7. `phase9 reconcile`, `phase9 audit`, `phase9 status`, and `phase9 freeze` publish the evidence trail.

No approval is reusable. No CLI command combines approval with submission.

## Public Artifact Privacy

Public artifacts may contain statuses, counts, hashes, and masked identifiers. They must not contain raw account IDs, broker order IDs, permIds, execIds, credentials, exact account values, exact cash values, or approval secrets.
