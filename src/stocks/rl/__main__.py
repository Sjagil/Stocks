from __future__ import annotations

import argparse
import json
from pathlib import Path

from stocks.p3.io import read_json
from stocks.rl.experience import ExperienceStore
from stocks.rl.registry import PolicyRegistry
from stocks.rl.supervisor import run_forever, run_supervisor_cycle
from stocks.rl.training import train_default_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shadow-only finance RL decision layer")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--timesteps", type=int)
    subparsers.add_parser("cycle")
    run = subparsers.add_parser("run")
    run.add_argument("--interval-seconds", type=int, default=900)
    run.add_argument("--disable-training", action="store_true")
    subparsers.add_parser("status")
    subparsers.add_parser("experience-status")
    verify = subparsers.add_parser("verify-policy")
    verify.add_argument("version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "train":
        result = train_default_experiment(root, total_timesteps=args.timesteps)
    elif args.command == "cycle":
        result = run_supervisor_cycle(root)
    elif args.command == "run":
        run_forever(
            root,
            interval_seconds=args.interval_seconds,
            allow_training=not args.disable_training,
        )
        return 0
    elif args.command == "experience-status":
        result = ExperienceStore(root).publish_status()
    elif args.command == "verify-policy":
        result = PolicyRegistry(root).verify(args.version)
    else:
        result = read_json(root / "output/rl/status.json") or {
            "status": "NOT_RUN",
            "rl_mode": "SHADOW_ONLY",
            "execution_authority": "NONE",
        }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") not in {"NO_GO", "BLOCKED", "FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
