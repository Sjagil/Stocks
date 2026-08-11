from __future__ import annotations

import json
from pathlib import Path

from stocks.ibkr.paper_execution.audit import privacy_audit


def test_project_env_and_process_environment_are_checked_independently(
    tmp_path: Path, monkeypatch,
) -> None:
    project_secret = "project-secret-123"
    process_secret = "process-secret-456"
    (tmp_path / ".env").write_text(
        f"EODHD_API_KEY={project_secret}\n", encoding="utf-8"
    )
    monkeypatch.setenv("EODHD_API_KEY", process_secret)
    output = tmp_path / "output/ibkr/phase9"
    output.mkdir(parents=True)
    (output / "safe.json").write_text(
        json.dumps({"api_key_configured": True}), encoding="utf-8"
    )

    result = privacy_audit(tmp_path)

    assert result["status"] == "GO"
    assert result["project_env_secret_values_checked"] == 1
    assert result["process_environment_secret_values_checked"] == 1
    assert result["distinct_secret_values_checked"] == 2
    assert project_secret not in json.dumps(result)
    assert process_secret not in json.dumps(result)


def test_secret_value_leaks_are_found_in_json_csv_and_logs(
    tmp_path: Path, monkeypatch,
) -> None:
    secret = "sensitive-value-123"
    monkeypatch.setenv("PROVIDER_SECRET", secret)
    output = tmp_path / "output/ibkr/phase9"
    output.mkdir(parents=True)
    (output / "leak.json").write_text(
        json.dumps({"value": secret}), encoding="utf-8"
    )
    (output / "leak.csv").write_text(
        f"name,value\nprovider,{secret}\n", encoding="utf-8"
    )
    (output / "leak.log").write_text(
        f"provider={secret}\n", encoding="utf-8"
    )

    result = privacy_audit(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["secret_leaks"] == 3
    assert secret not in json.dumps(result)


def test_raw_account_identifier_is_blocking(tmp_path: Path) -> None:
    output = tmp_path / "output/ibkr/phase9"
    output.mkdir(parents=True)
    (output / "account.json").write_text(
        '{"account":"DU1234567"}', encoding="utf-8"
    )

    result = privacy_audit(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["account_leaks"] == 1

