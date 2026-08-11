# Active Swing Sprints 3-6

This layer extends the frozen setup and forward-episode pipeline. It does not
change strategy or execution authority.

## Sprint 3: bounded shortlist data

The funnel is capped at 20 structural candidates, 10 Level I/tape candidates
and 5 temporary depth candidates. Every component is classified as one of:

```text
AVAILABLE_LIVE
AVAILABLE_DELAYED
AVAILABLE_BAR_PROXY
UNAVAILABLE
STALE
```

Bar-derived volume is never represented as tape, CVD or order-book evidence.
Missing subscriptions and private stores remain explicitly unavailable.

## Sprint 4: entry-filter experiment

The experiment compares exactly four treatments over identical immutable
episodes:

```text
BASE
LEVEL1_TAPE
TAPE_DEPTH
ASSET_PROFILE
```

It reuses the same setup, entry, stop, targets and terminal outcome. It does
not mutate an episode or infer an intrabar path. Promotion requires at least
50 closed independent episodes and positive incremental evidence after costs.

## Sprint 5: role leaderboards

Leaderboards are separated into:

```text
STRATEGIC_ALLOCATION
ACTIVE_SWING
TACTICAL_ENTRY
EVENT_DRIVEN
COMMODITY_PROXY
```

Each role publishes at most one champion and two challengers. Small samples
are shrunk toward a neutral score. Cross-role ranking and automatic promotion
are prohibited.

## Sprint 6: selective ML

ML is an observation-only meta-labeler. Logistic regression is the baseline
and histogram gradient boosting is the challenger. Training is only started
by an explicit CLI command.

```text
<150 labels     not trained
150-499         experimental report only
500-999         shadow comparison eligible
1000+           paper-ranking research eligible
```

No model has order authority. Runtime refreshes data coverage, experiment
metrics and leaderboards, but never retrains a model automatically.

## Commands

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py research active-swing shortlist-data
.\.venv-ibkr\Scripts\python.exe .\main.py research active-swing entry-filter-experiment
.\.venv-ibkr\Scripts\python.exe .\main.py research active-swing leaderboards
.\.venv-ibkr\Scripts\python.exe .\main.py research active-swing train-ml
.\.venv-ibkr\Scripts\python.exe .\main.py research active-swing refresh
.\.venv-ibkr\Scripts\python.exe .\main.py research active-swing status
```

`refresh` is the canonical runtime command and deliberately excludes model
training. `run` is an explicit operator command that also evaluates the ML
training gate.

All outputs are written below:

```text
output/research/active_swing/
```
