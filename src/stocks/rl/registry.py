from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.p3.io import atomic_write_json, file_hash, read_json
from stocks.rl.contracts import PolicyState, stable_hash


REQUIRED_POLICY_FILES = (
    "model.zip",
    "config.json",
    "reward_config.json",
    "feature_schema.json",
    "training_metadata.json",
    "evaluation.json",
    "regime_performance.json",
    "cost_stress.json",
    "policy_hash.txt",
)


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    passed: bool
    reasons: tuple[str, ...]
    evidence_hash: str
    challenger_version: str
    active_version: str | None
    safety_regression: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reasons": list(self.reasons),
            "challenger_cannot_self_promote": True,
            "rl_live_enabled": False,
            "execution_authority": "NONE",
        }


class PolicyRegistry:
    def __init__(self, project_root: Path, root: Path = Path("models/rl")) -> None:
        self.project_root = project_root.resolve()
        self.root = (self.project_root / root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        if not self.registry_path.exists():
            self._write(
                {
                    "schema": "rl_policy_registry_v1",
                    "active": None,
                    "challenger": None,
                    "policies": {},
                    "automatic_promotion": False,
                    "rl_live_enabled": False,
                    "execution_authority": "NONE",
                    "money_control": False,
                    "updated_at": _now(),
                }
            )

    def register(
        self,
        version: str,
        *,
        model_source: Path,
        config: dict[str, Any],
        reward_config: dict[str, Any],
        feature_schema: dict[str, Any],
        training_metadata: dict[str, Any],
        evaluation: dict[str, Any],
        regime_performance: dict[str, Any],
        cost_stress: dict[str, Any],
        state: PolicyState = PolicyState.CHALLENGER,
    ) -> dict[str, Any]:
        if not version or any(character in version for character in "\\/:*?\"<>|"):
            raise ValueError("invalid RL policy version")
        if state is PolicyState.ACTIVE:
            raise ValueError("new RL policies cannot self-register as ACTIVE")
        if not model_source.is_file():
            raise ValueError("RL model artifact missing")
        registry = self.read()
        existing = registry["policies"].get(version)
        if existing:
            if existing.get("model_hash") != file_hash(model_source):
                raise ValueError("immutable RL policy version collision")
            return existing

        destination = self.root / version
        destination.mkdir(parents=False, exist_ok=False)
        model_path = destination / "model.zip"
        model_path.write_bytes(model_source.read_bytes())
        metadata = {
            "config.json": config,
            "reward_config.json": reward_config,
            "feature_schema.json": feature_schema,
            "training_metadata.json": training_metadata,
            "evaluation.json": evaluation,
            "regime_performance.json": regime_performance,
            "cost_stress.json": cost_stress,
        }
        for name, payload in metadata.items():
            atomic_write_json(destination / name, payload)
        policy_hash = stable_hash(
            {
                "model_hash": file_hash(model_path),
                "metadata_hashes": {
                    name: file_hash(destination / name) for name in sorted(metadata)
                },
            }
        )
        (destination / "policy_hash.txt").write_text(
            policy_hash + "\n", encoding="utf-8"
        )
        record = {
            "version": version,
            "state": state.value,
            "path": destination.relative_to(self.project_root).as_posix(),
            "model_hash": file_hash(model_path),
            "policy_hash": policy_hash,
            "registered_at": _now(),
            "training_data_end": training_metadata.get("training_end"),
            "evaluation_status": evaluation.get("status"),
            "promotion_eligible": False,
            "execution_authority": "NONE",
            "money_control": False,
        }
        registry["policies"][version] = record
        if state is PolicyState.CHALLENGER:
            previous = registry.get("challenger")
            if previous and previous in registry["policies"]:
                registry["policies"][previous]["state"] = PolicyState.ARCHIVED.value
            registry["challenger"] = version
        registry["updated_at"] = _now()
        self._write(registry)
        return record

    def record_promotion_decision(
        self, decision: PromotionDecision
    ) -> dict[str, Any]:
        registry = self.read()
        if decision.challenger_version not in registry["policies"]:
            raise ValueError("promotion decision references unknown challenger")
        path = self.root / decision.challenger_version / "promotion-decision.json"
        payload = decision.to_dict()
        atomic_write_json(path, payload)
        registry["policies"][decision.challenger_version][
            "promotion_eligible"
        ] = bool(decision.passed and not decision.safety_regression)
        registry["policies"][decision.challenger_version][
            "promotion_decision_hash"
        ] = file_hash(path)
        registry["updated_at"] = _now()
        self._write(registry)
        return payload

    def promote(
        self,
        version: str,
        *,
        operator_approved: bool,
        approval_text: str,
    ) -> dict[str, Any]:
        if not operator_approved or approval_text != "PROMOTE RL SHADOW POLICY":
            raise PermissionError("explicit RL shadow promotion approval required")
        registry = self.read()
        record = registry["policies"].get(version)
        if not record:
            raise ValueError("unknown RL policy version")
        decision = read_json(self.root / version / "promotion-decision.json")
        if not decision.get("passed") or decision.get("safety_regression"):
            raise PermissionError("RL challenger did not pass the promotion gate")
        current = registry.get("active")
        if current and current in registry["policies"]:
            registry["policies"][current]["state"] = PolicyState.ARCHIVED.value
        record["state"] = PolicyState.ACTIVE.value
        record["activated_at"] = _now()
        record["activation_scope"] = "SHADOW_INFERENCE_ONLY"
        registry["active"] = version
        registry["challenger"] = None
        registry["updated_at"] = _now()
        self._write(registry)
        return record

    def reject(self, version: str, reasons: list[str]) -> dict[str, Any]:
        registry = self.read()
        record = registry["policies"].get(version)
        if not record:
            raise ValueError("unknown RL policy version")
        record["state"] = PolicyState.REJECTED.value
        record["rejection_reasons"] = list(reasons)
        record["rejected_at"] = _now()
        if registry.get("challenger") == version:
            registry["challenger"] = None
        registry["updated_at"] = _now()
        self._write(registry)
        return record

    def verify(self, version: str) -> dict[str, Any]:
        registry = self.read()
        record = registry.get("policies", {}).get(version)
        if not record:
            return {"status": "NO_GO", "blockers": ["POLICY_NOT_REGISTERED"]}
        directory = self.project_root / record["path"]
        missing = [name for name in REQUIRED_POLICY_FILES if not (directory / name).is_file()]
        calculated = None
        if not missing:
            metadata_names = [
                name for name in REQUIRED_POLICY_FILES if name not in {"model.zip", "policy_hash.txt"}
            ]
            calculated = stable_hash(
                {
                    "model_hash": file_hash(directory / "model.zip"),
                    "metadata_hashes": {
                        name: file_hash(directory / name) for name in sorted(metadata_names)
                    },
                }
            )
        expected = (
            (directory / "policy_hash.txt").read_text(encoding="utf-8").strip()
            if (directory / "policy_hash.txt").is_file()
            else None
        )
        blockers = list(missing)
        if calculated != expected or expected != record.get("policy_hash"):
            blockers.append("POLICY_HASH_MISMATCH")
        return {
            "status": "GO" if not blockers else "NO_GO",
            "version": version,
            "state": record.get("state"),
            "blockers": blockers,
            "policy_hash": expected,
            "execution_authority": "NONE",
            "money_control": False,
        }

    def read(self) -> dict[str, Any]:
        payload = read_json(self.registry_path)
        if payload.get("schema") != "rl_policy_registry_v1":
            raise ValueError("invalid RL policy registry")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        payload["registry_hash"] = stable_hash(
            {key: value for key, value in payload.items() if key != "registry_hash"}
        )
        atomic_write_json(self.registry_path, payload)


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "PolicyRegistry",
    "PromotionDecision",
    "REQUIRED_POLICY_FILES",
]
