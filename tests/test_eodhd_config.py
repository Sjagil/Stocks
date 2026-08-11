from __future__ import annotations

from stocks.data.eodhd import load_eodhd_settings


def test_eodhd_settings_default_disabled_without_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    monkeypatch.delenv("EODHD_ENABLED", raising=False)

    settings = load_eodhd_settings(tmp_path / ".env")

    assert settings.enabled is False
    assert settings.safe_dict() == {
        "provider": "EODHD",
        "enabled": False,
        "requested_enabled": False,
        "api_key_configured": False,
        "authority": "disabled_until_data_phase",
        "status": "DISABLED",
    }


def test_eodhd_settings_request_with_key_stays_disabled_until_data_phase(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    monkeypatch.delenv("EODHD_ENABLED", raising=False)
    (tmp_path / ".env").write_text(
        "EODHD_API_KEY=secret-value\nEODHD_ENABLED=true\n",
        encoding="utf-8",
    )

    settings = load_eodhd_settings(tmp_path / ".env")

    assert settings.api_key_configured is True
    assert settings.requested_enabled is True
    assert settings.enabled is False
    assert settings.safe_dict()["status"] == "DISABLED_UNTIL_DATA_PHASE"


def test_eodhd_settings_can_only_enable_with_explicit_data_phase_authority(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EODHD_API_KEY", "secret-value")
    monkeypatch.setenv("EODHD_ENABLED", "yes")

    settings = load_eodhd_settings(tmp_path / ".env", data_phase_enabled=True)

    assert settings.api_key_configured is True
    assert settings.requested_enabled is True
    assert settings.enabled is True
    assert settings.safe_dict() == {
        "provider": "EODHD",
        "enabled": True,
        "requested_enabled": True,
        "api_key_configured": True,
        "authority": "research_data_read_only",
        "status": "GO",
    }
