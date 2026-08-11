# Phase 9 Status

phase9_status: PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO
phase9_marker: PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO
execution_authority: MANUAL_PAPER_CANARY
target_execution_authority_after_full_evidence: MANUAL_PAPER_CANARY
strategy_authority: NONE
shadow_authority: NONE
live_authority: NONE
manual_approval_required: true
automatic_submission: false
paper_only: true
financial_decision: NO_NEW_FINANCIAL_CANDIDATE
FINANCIAL_FINALIST_GO: false
PAPER_STRATEGY_AUTHORITY: blocked
LIVE_STRATEGY_AUTHORITY: blocked

## Checks

- audit: True
- closing_sell_canary: True
- fill_canary: True
- phase8_2_freeze: True
- preflight: True
- reconciliation: True
- schema: True
- submit_cancel_canary: True

## Open Blockers

- none

## Counters

paper_place_order_calls: 0
paper_cancel_order_calls: 0
live_place_order_calls: 0
global_cancel_calls: 0
market_data_calls: 0
historical_data_calls: 0
strategy_generated_intents: 0
automatic_submissions: 0
automatic_cancellations: 0

## Current Safe Operator Sequence

1. Keep TWS Paper open and logged in.
2. Configure only .env.ibkr paper-writer fields.
3. Run: python .\main.py ibkr phase9 preflight
4. Only if preflight is GO, run prepare, approve, submit, reconcile, audit, status.
5. Run cancel canary with a separate prepare-cancel and approve-cancel step.
6. Run freeze only after paper canary evidence exists.

No strategy-generated intent, live order, global cancel, market data, or historical data call is authorized by Phase 9.
