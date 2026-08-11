from __future__ import annotations

import json
from pathlib import Path

import pytest

import main
from stocks.application.config import IbkrSettings, load_ibkr_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IBKR_ENV_KEYS = (
    "APP_ENV",
    "IBKR_ACCOUNT",
    "IBKR_HOST",
    "IBKR_PORT",
    "IBKR_CLIENT_ID",
    "IBKR_READ_ONLY",
    "IBKR_ORDER_AUTHORITY",
    "IBKR_LIVE_TRADING_ENABLED",
    "IBKR_ALLOW_ORDER_TRANSMISSION",
    "IBKR_MARKET_DATA_TYPE",
    "IBKR_ALLOWED_SECURITY_TYPES",
    "IBKR_ALLOWED_CURRENCIES",
    "IBKR_MAX_ORDER_NOTIONAL_EUR",
    "IBKR_MAX_OPEN_ORDERS",
    "IBKR_MAX_POSITIONS",
    "IBKR_CONNECT_TIMEOUT_SECONDS",
    "IBKR_REQUEST_TIMEOUT_SECONDS",
    "IBKR_HEARTBEAT_INTERVAL_SECONDS",
    "IBKR_STALE_AFTER_SECONDS",
    "IBKR_MAX_RECONNECT_ATTEMPTS",
    "IBKR_RECONNECT_DELAYS_SECONDS",
    "IBKR_OUTPUT_DIR",
    "IBKR_LOG_LEVEL",
)


def test_loads_safe_ibkr_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = load_ibkr_settings(Path(".env.ibkr"))

    assert settings.host == "127.0.0.1"
    assert settings.port == 7497
    assert settings.client_id == 17
    assert settings.read_only is True
    assert settings.order_authority == "NONE"
    assert settings.live_trading_enabled is False
    assert settings.allow_order_transmission is False
    assert settings.max_order_notional_eur == 0
    assert settings.max_open_orders == 0
    assert settings.max_positions == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("IBKR_READ_ONLY", "tru"),
        ("IBKR_LIVE_TRADING_ENABLED", "maybe"),
        ("IBKR_ALLOW_ORDER_TRANSMISSION", "disabled"),
    ],
)
def test_load_ibkr_settings_rejects_invalid_boolean_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(f"{name}={value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid boolean value"):
        load_ibkr_settings(env_file)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("IBKR_PORT", " 7497", "invalid integer value"),
        ("IBKR_CLIENT_ID", "17.0", "invalid integer value"),
        ("IBKR_MAX_OPEN_ORDERS", "false", "invalid integer value"),
        ("IBKR_CONNECT_TIMEOUT_SECONDS", " 12.0", "invalid float value"),
        ("IBKR_REQUEST_TIMEOUT_SECONDS", "nan", "invalid float value"),
        ("IBKR_REQUEST_TIMEOUT_SECONDS", "1e2", "invalid float value"),
        ("IBKR_HEARTBEAT_INTERVAL_SECONDS", "true", "invalid float value"),
        ("IBKR_STALE_AFTER_SECONDS", "[45]", "invalid float value"),
        ("IBKR_MAX_RECONNECT_ATTEMPTS", "0.0", "invalid integer value"),
        ("IBKR_RECONNECT_DELAYS_SECONDS", "2, 5", "invalid float value"),
    ],
)
def test_load_ibkr_settings_rejects_invalid_numeric_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        load_ibkr_settings(tmp_path / ".env.ibkr")


def test_load_ibkr_settings_reads_bounded_reconnect_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(
        "\n".join(
            [
                "IBKR_HEARTBEAT_INTERVAL_SECONDS=10",
                "IBKR_STALE_AFTER_SECONDS=30",
                "IBKR_MAX_RECONNECT_ATTEMPTS=3",
                "IBKR_RECONNECT_DELAYS_SECONDS=1,2.5,5",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_ibkr_settings(env_file)

    assert settings.heartbeat_interval_seconds == 10.0
    assert settings.stale_after_seconds == 30.0
    assert settings.max_reconnect_attempts == 3
    assert settings.reconnect_delays_seconds == (1.0, 2.5, 5.0)


@pytest.mark.parametrize("env_file_name", [".env.ibkr.example", "env.ibkr.example"])
def test_public_ibkr_env_examples_parse(env_file_name: str) -> None:
    settings = load_ibkr_settings(PROJECT_ROOT / env_file_name)

    assert settings.app_env == "development"
    assert settings.host == "127.0.0.1"
    assert settings.port == 7497
    assert settings.market_data_type == 3
    assert settings.read_only is True
    assert settings.order_authority == "NONE"
    assert settings.reconnect_delays_seconds == (2.0, 5.0, 15.0, 30.0)
    assert settings.log_level == "INFO"


def test_load_ibkr_settings_canonicalizes_app_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text("APP_ENV=PAPER\n", encoding="utf-8")

    settings = load_ibkr_settings(env_file)

    assert settings.app_env == "paper"


@pytest.mark.parametrize("app_env", ["production", "live", "staging", " development"])
def test_load_ibkr_settings_rejects_unsafe_app_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_env: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APP_ENV", app_env)

    with pytest.raises(ValueError, match="APP_ENV"):
        load_ibkr_settings(tmp_path / ".env.ibkr")


def test_rejects_unsafe_app_env() -> None:
    with pytest.raises(ValueError, match="APP_ENV"):
        IbkrSettings(app_env="production")


def test_load_ibkr_settings_canonicalizes_localhost_to_freeze_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text("IBKR_HOST=localhost\n", encoding="utf-8")

    settings = load_ibkr_settings(env_file)

    assert settings.host == "127.0.0.1"


@pytest.mark.parametrize("host", ["::1", "127.0.0.1 ", "192.168.1.10"])
def test_load_ibkr_settings_rejects_noncanonical_phase1_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IBKR_HOST", host)

    with pytest.raises(ValueError, match="IBKR_HOST"):
        load_ibkr_settings(tmp_path / ".env.ibkr")


def test_direct_settings_canonicalize_localhost_to_freeze_host() -> None:
    settings = IbkrSettings(host="localhost")

    assert settings.host == "127.0.0.1"


def test_load_ibkr_settings_canonicalizes_order_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text("IBKR_ORDER_AUTHORITY=none\n", encoding="utf-8")

    settings = load_ibkr_settings(env_file)

    assert settings.order_authority == "NONE"


@pytest.mark.parametrize("order_authority", ["PAPER", "LIVE", " NONE", "NONE "])
def test_load_ibkr_settings_rejects_invalid_order_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order_authority: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IBKR_ORDER_AUTHORITY", order_authority)

    with pytest.raises(ValueError, match="IBKR_ORDER_AUTHORITY"):
        load_ibkr_settings(tmp_path / ".env.ibkr")


def test_direct_settings_canonicalize_order_authority() -> None:
    settings = IbkrSettings(order_authority="none")

    assert settings.order_authority == "NONE"


def test_load_ibkr_settings_canonicalizes_log_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text("IBKR_LOG_LEVEL=debug\n", encoding="utf-8")

    settings = load_ibkr_settings(env_file)

    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("log_level", ["TRACE", "WARN", " INFO", "DEBUG "])
def test_load_ibkr_settings_rejects_invalid_log_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_level: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IBKR_LOG_LEVEL", log_level)

    with pytest.raises(ValueError, match="IBKR_LOG_LEVEL"):
        load_ibkr_settings(tmp_path / ".env.ibkr")


def test_rejects_invalid_log_level() -> None:
    with pytest.raises(ValueError, match="IBKR_LOG_LEVEL"):
        IbkrSettings(log_level="VERBOSE")


def test_load_ibkr_settings_requires_empty_account_in_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(f"IBKR_ACCOUNT={'DU' + '0000000'}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IBKR_ACCOUNT must remain empty"):
        load_ibkr_settings(env_file)


def test_load_ibkr_settings_requires_empty_account_in_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IBKR_ACCOUNT", "U" + "0000000")

    with pytest.raises(ValueError, match="IBKR_ACCOUNT must remain empty"):
        load_ibkr_settings(tmp_path / ".env.ibkr")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connect_timeout_seconds": float("nan")},
        {"request_timeout_seconds": float("inf")},
        {"heartbeat_interval_seconds": 45.0, "stale_after_seconds": 45.0},
        {"max_reconnect_attempts": 0},
        {"reconnect_delays_seconds": ()},
        {"reconnect_delays_seconds": (0.0, 1.0)},
        {"reconnect_delays_seconds": (1.0, float("inf"))},
    ],
)
def test_rejects_invalid_timing_and_reconnect_policy(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IbkrSettings(**kwargs)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("IBKR_STALE_AFTER_SECONDS=25\nIBKR_HEARTBEAT_INTERVAL_SECONDS=25", "STALE"),
        ("IBKR_MAX_RECONNECT_ATTEMPTS=0", "MAX_RECONNECT_ATTEMPTS"),
        ("IBKR_RECONNECT_DELAYS_SECONDS=,,,", "RECONNECT_DELAYS"),
        ("IBKR_RECONNECT_DELAYS_SECONDS=1,0,5", "RECONNECT_DELAYS"),
    ],
)
def test_load_ibkr_settings_rejects_invalid_reconnect_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line: str,
    message: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_ibkr_settings(env_file)


@pytest.mark.parametrize("market_data_type", [1, 2, 4])
def test_rejects_non_delayed_market_data_type(market_data_type: int) -> None:
    with pytest.raises(ValueError, match="IBKR_MARKET_DATA_TYPE=3"):
        IbkrSettings(market_data_type=market_data_type)


def test_load_ibkr_settings_rejects_live_market_data_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text("IBKR_MARKET_DATA_TYPE=1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IBKR_MARKET_DATA_TYPE=3"):
        load_ibkr_settings(env_file)


@pytest.mark.parametrize("output_dir", ["output/ibkr", "output\\ibkr"])
def test_load_ibkr_settings_accepts_canonical_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_dir: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(f"IBKR_OUTPUT_DIR={output_dir}\n", encoding="utf-8")

    settings = load_ibkr_settings(env_file)

    assert settings.output_dir == Path("output/ibkr")


@pytest.mark.parametrize(
    "output_dir",
    [
        "output",
        "output/ibkr/contracts",
        "../output/ibkr",
        "C:/Users/alhar/Documents/Stocks/output/ibkr",
    ],
)
def test_load_ibkr_settings_rejects_noncanonical_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_dir: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(f"IBKR_OUTPUT_DIR={output_dir}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IBKR_OUTPUT_DIR"):
        load_ibkr_settings(env_file)


def test_load_ibkr_settings_rejects_output_dir_env_override_whitespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IBKR_OUTPUT_DIR", " output/ibkr")

    with pytest.raises(ValueError, match="IBKR_OUTPUT_DIR"):
        load_ibkr_settings(tmp_path / ".env.ibkr")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_security_types": ()},
        {"allowed_security_types": ("STK", "OPT")},
        {"allowed_currencies": ()},
        {"allowed_currencies": ("USD", "JPY")},
    ],
)
def test_rejects_security_type_and_currency_scope_expansion(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IbkrSettings(**kwargs)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("IBKR_ALLOWED_SECURITY_TYPES=STK,OPT", "IBKR_ALLOWED_SECURITY_TYPES"),
        ("IBKR_ALLOWED_SECURITY_TYPES=,,,", "IBKR_ALLOWED_SECURITY_TYPES"),
        ("IBKR_ALLOWED_CURRENCIES=USD,JPY", "IBKR_ALLOWED_CURRENCIES"),
        ("IBKR_ALLOWED_CURRENCIES=,,,", "IBKR_ALLOWED_CURRENCIES"),
    ],
)
def test_load_ibkr_settings_rejects_allowlist_scope_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line: str,
    message: str,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_ibkr_settings(env_file)


def test_load_ibkr_settings_canonicalizes_valid_allowlists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(
        "\n".join(
            [
                "IBKR_ALLOWED_SECURITY_TYPES=stk,fut",
                "IBKR_ALLOWED_CURRENCIES=eur,usd",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_ibkr_settings(env_file)

    assert settings.allowed_security_types == ("STK", "FUT")
    assert settings.allowed_currencies == ("EUR", "USD")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 7496},
        {"port": 4001},
        {"host": "192.168.1.10"},
        {"read_only": False},
        {"order_authority": "PAPER"},
        {"live_trading_enabled": True},
        {"allow_order_transmission": True},
        {"max_order_notional_eur": 1},
        {"max_open_orders": 1},
        {"max_positions": 1},
    ],
)
def test_rejects_non_read_only_authority(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IbkrSettings(**kwargs)


def test_safe_dict_does_not_emit_account_identifier() -> None:
    account = "DU" + "1234567"
    settings = IbkrSettings(account=account)
    safe = settings.safe_dict()

    assert safe["account_configured"] is True
    assert account not in str(safe)


def test_disconnect_drill_preflight_reports_go_for_safe_tws_paper_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text(
        "\n".join(
            [
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7497",
                "IBKR_CLIENT_ID=17",
                "IBKR_READ_ONLY=true",
                "IBKR_ORDER_AUTHORITY=NONE",
                "IBKR_LIVE_TRADING_ENABLED=false",
                "IBKR_ALLOW_ORDER_TRANSMISSION=false",
                "IBKR_MAX_ORDER_NOTIONAL_EUR=0",
                "IBKR_MAX_OPEN_ORDERS=0",
                "IBKR_MAX_POSITIONS=0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main.main(
        [
            "--env-file",
            str(env_file),
            "ibkr",
            "disconnect-drill-preflight",
            "--skip-socket-check",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "phase1_disconnect_drill_preflight_v1"
    assert payload["status"] == "GO"
    assert payload["checks"]["tws_paper_host_127_0_0_1"] is True
    assert payload["checks"]["tws_paper_port_7497"] is True
    assert payload["checks"]["paper_socket_reachable"] is None
    assert payload["financial_calls"]["place_order"] == 0


def test_disconnect_drill_preflight_rejects_gateway_port_for_tws_drill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in IBKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env.ibkr"
    env_file.write_text("IBKR_PORT=4002\n", encoding="utf-8")

    exit_code = main.main(
        [
            "--env-file",
            str(env_file),
            "ibkr",
            "disconnect-drill-preflight",
            "--skip-socket-check",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "NO_GO"
    assert payload["checks"]["tws_paper_port_7497"] is False
    assert payload["blocking_checks"] == ["tws_paper_port_7497"]
