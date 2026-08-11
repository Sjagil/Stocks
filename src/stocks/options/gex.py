from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd


class GexInputError(ValueError):
    pass


ALIASES = {
    "expiration": ("expiration", "expiry", "expiration_date"),
    "option_type": ("option_type", "type", "right", "put_call"),
    "strike": ("strike", "strike_price"),
    "open_interest": ("open_interest", "openinterest", "oi"),
    "implied_volatility": (
        "implied_volatility",
        "impliedvolatility",
        "iv",
    ),
    "contract_multiplier": (
        "contract_multiplier",
        "multiplier",
        "contract_size",
    ),
    "gamma": ("gamma", "option_gamma"),
}


def gex_schema() -> dict[str, Any]:
    return {
        "schema": "stocks_gex_context_v1",
        "status": "GO",
        "authority": "CONTEXT_ONLY",
        "execution_authority": "NONE",
        "dealer_position_observed": False,
        "formula": "gamma * open_interest * multiplier * spot^2 * 0.01",
        "required_chain_fields": [
            "expiration",
            "option_type",
            "strike",
            "open_interest",
        ],
        "gamma_inputs": [
            "provider gamma",
            "or implied volatility for Black-Scholes gamma",
        ],
        "dealer_sign_proxy": "calls positive, puts negative",
        "limitations": [
            "dealer inventory direction is estimated, not observed",
            "a current option chain is not historical point-in-time data",
            "GEX is timing and sizing context, never standalone authority",
        ],
    }


def calculate_gex_snapshot(
    chain: pd.DataFrame,
    *,
    symbol: str,
    spot: float,
    as_of: datetime | pd.Timestamp,
    source: str,
    source_mode: str = "CURRENT_CHAIN_NOT_PIT",
    risk_free_rate: float = 0.03,
    dividend_yield: float = 0.0,
    max_age_hours: float = 24.0,
    observed_at: datetime | pd.Timestamp | None = None,
    atr_1d: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not math.isfinite(float(spot)) or float(spot) <= 0:
        raise GexInputError("spot must be positive and finite")
    if chain.empty:
        raise GexInputError("option chain is empty")

    as_of_ts = _utc_timestamp(as_of)
    observed_ts = _utc_timestamp(observed_at or datetime.now(UTC))
    normalized, quality = _normalize_chain(chain, as_of_ts)
    if normalized.empty:
        raise GexInputError("no valid option rows remain after validation")

    strike = normalized["strike"].to_numpy(dtype=float)
    years = normalized["time_to_expiry_years"].to_numpy(dtype=float)
    open_interest = normalized["open_interest"].to_numpy(dtype=float)
    multiplier = normalized["contract_multiplier"].to_numpy(dtype=float)
    gamma, gamma_provenance = _gamma_values(
        normalized,
        spot=float(spot),
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
    )
    sign = np.where(
        normalized["option_type"].to_numpy(dtype=str) == "call",
        1.0,
        -1.0,
    )
    signed_gex = (
        sign
        * gamma
        * open_interest
        * multiplier
        * float(spot) ** 2
        * 0.01
    )
    normalized["gamma_used"] = gamma
    normalized["gamma_provenance"] = gamma_provenance
    normalized["signed_gex_1pct"] = signed_gex
    normalized["call_gex_1pct"] = np.where(sign > 0, signed_gex, 0.0)
    normalized["put_gex_1pct"] = np.where(sign < 0, signed_gex, 0.0)

    profile = (
        normalized.groupby("strike", as_index=False)
        .agg(
            signed_gex_1pct=("signed_gex_1pct", "sum"),
            call_gex_1pct=("call_gex_1pct", "sum"),
            put_gex_1pct=("put_gex_1pct", "sum"),
            open_interest=("open_interest", "sum"),
            option_rows=("strike", "size"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )
    call_wall = float(profile.loc[profile["call_gex_1pct"].idxmax(), "strike"])
    put_wall = float(profile.loc[profile["put_gex_1pct"].idxmin(), "strike"])
    scenario = _gex_scenario(
        strike=strike,
        years=years,
        open_interest=open_interest,
        multiplier=multiplier,
        implied_volatility=normalized["implied_volatility"].to_numpy(
            dtype=float
        ),
        observed_gamma=normalized["gamma"].to_numpy(dtype=float),
        use_observed_gamma=normalized["gamma"].notna().to_numpy(),
        sign=sign,
        spot=float(spot),
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
    )
    gamma_flip = _nearest_zero_crossing(scenario, float(spot))

    absolute = profile["signed_gex_1pct"].abs()
    gross_gex = float(absolute.sum())
    top_n = min(3, len(profile))
    concentration = (
        float(absolute.nlargest(top_n).sum() / gross_gex)
        if gross_gex > 0
        else 0.0
    )
    dte = normalized["time_to_expiry_years"] * 365.25
    gross_rows = normalized["signed_gex_1pct"].abs()
    near_expiry_concentration = _safe_share(
        gross_rows[dte <= 7.0].sum(), gross_rows.sum()
    )
    zero_dte_concentration = _safe_share(
        gross_rows[dte < 1.0].sum(), gross_rows.sum()
    )
    age_hours = max(
        0.0,
        (observed_ts - as_of_ts).total_seconds() / 3600.0,
    )
    freshness = max(0.0, 1.0 - age_hours / max(max_age_hours, 1e-9))
    gamma_quality = (
        float((normalized["gamma_provenance"] == "PROVIDER_OBSERVED").mean())
        * 1.0
        + float(
            (normalized["gamma_provenance"] == "BLACK_SCHOLES_ESTIMATED").mean()
        )
        * 0.7
    )
    breadth = min(1.0, len(normalized) / 500.0)
    expiration_breadth = min(
        1.0,
        normalized["expiration"].dt.date.nunique() / 4.0,
    )
    confidence = (
        0.30 * freshness
        + 0.25 * quality["valid_row_ratio"]
        + 0.20 * gamma_quality
        + 0.15 * breadth
        + 0.10 * expiration_breadth
    )
    if source_mode != "CERTIFIED_POINT_IN_TIME_CHAIN":
        confidence = min(confidence, 0.70)

    net_gex = float(signed_gex.sum())
    summary = {
        "schema": "stocks_gex_context_v1",
        "status": "AVAILABLE_CONTEXT_ONLY"
        if freshness > 0
        else "STALE_CONTEXT_BLOCKED",
        "symbol": str(symbol).upper(),
        "as_of": as_of_ts.isoformat(),
        "observed_at": observed_ts.isoformat(),
        "age_hours": round(age_hours, 6),
        "spot": float(spot),
        "source": str(source),
        "source_mode": str(source_mode),
        "source_is_point_in_time": source_mode
        == "CERTIFIED_POINT_IN_TIME_CHAIN",
        "net_gex_1pct": net_gex,
        "gross_absolute_gex_1pct": gross_gex,
        "regime_proxy": _gex_regime(net_gex, gross_gex),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "distance_to_call_wall_atr": _atr_distance(
            float(spot), call_wall, atr_1d
        ),
        "distance_to_put_wall_atr": _atr_distance(
            float(spot), put_wall, atr_1d
        ),
        "distance_to_flip_atr": _atr_distance(
            float(spot), gamma_flip, atr_1d
        ),
        "gex_concentration_top3": round(concentration, 8),
        "near_expiry_gex_concentration": round(
            near_expiry_concentration, 8
        ),
        "zero_dte_gex_concentration": round(zero_dte_concentration, 8),
        "expiration_count": int(
            normalized["expiration"].dt.date.nunique()
        ),
        "option_rows": int(len(normalized)),
        "raw_option_rows": int(quality["raw_rows"]),
        "valid_row_ratio": round(quality["valid_row_ratio"], 8),
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 8),
        "confidence_class": _confidence_class(confidence),
        "gamma_provenance_counts": {
            str(key): int(value)
            for key, value in normalized["gamma_provenance"]
            .value_counts()
            .to_dict()
            .items()
        },
        "observed_data": [
            "expiration",
            "option_type",
            "strike",
            "open_interest",
            "implied_volatility where supplied",
            "gamma where supplied",
        ],
        "estimated_data": [
            "dealer direction proxy",
            "Black-Scholes gamma where provider gamma is absent",
            "gamma flip scenario",
        ],
        "dealer_position_observed": False,
        "dealer_sign_proxy": "calls positive, puts negative",
        "authority": "CONTEXT_ONLY",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "warnings": [
            "CURRENT_CHAIN_NOT_HISTORICAL_PIT"
            if source_mode != "CERTIFIED_POINT_IN_TIME_CHAIN"
            else "",
            "DEALER_POSITION_DIRECTION_IS_A_PROXY",
        ],
    }
    summary["warnings"] = [item for item in summary["warnings"] if item]
    return summary, profile, scenario


def adapt_external_gex_snapshot(
    payload: Mapping[str, Any],
    *,
    observed_at: datetime | pd.Timestamp | None = None,
    max_age_hours: float = 48.0,
) -> list[dict[str, Any]]:
    generated_at = _utc_timestamp(
        payload.get("generated_at") or datetime.fromtimestamp(0, UTC)
    )
    observed_ts = _utc_timestamp(observed_at or datetime.now(UTC))
    age_hours = max(
        0.0,
        (observed_ts - generated_at).total_seconds() / 3600.0,
    )
    rows: list[dict[str, Any]] = []
    underlyings = payload.get("equity_etf_underlyings")
    if not isinstance(underlyings, Mapping):
        return rows
    for symbol, raw in sorted(underlyings.items()):
        if not isinstance(raw, Mapping):
            continue
        walls = raw.get("gamma_walls")
        walls = walls if isinstance(walls, Mapping) else {}
        positive = walls.get("positive_gamma_wall")
        negative = walls.get("negative_gamma_wall")
        positive = positive if isinstance(positive, Mapping) else {}
        negative = negative if isinstance(negative, Mapping) else {}
        fresh = age_hours <= max_age_hours
        rows.append(
            {
                "schema": "stocks_external_gex_context_adapter_v1",
                "status": (
                    "AVAILABLE_CONTEXT_ONLY" if fresh else "STALE_CONTEXT_BLOCKED"
                ),
                "symbol": str(symbol).upper(),
                "as_of": generated_at.isoformat(),
                "observed_at": observed_ts.isoformat(),
                "age_hours": round(age_hours, 6),
                "source": raw.get("source_id", "UNKNOWN_EXTERNAL_SOURCE"),
                "source_mode": "CURRENT_CHAIN_ANALYTICS_NOT_PIT",
                "source_is_point_in_time": False,
                "net_gex_legacy": _finite_or_none(raw.get("GEX")),
                "net_gex_1pct": None,
                "regime_proxy": _sign_regime(raw.get("GEX")),
                "call_wall": _finite_or_none(positive.get("strike")),
                "put_wall": _finite_or_none(negative.get("strike")),
                "gamma_flip": None,
                "open_interest": _finite_or_none(raw.get("open_interest")),
                "iv_skew_25d": _finite_or_none(raw.get("iv_skew_25d")),
                "confidence": 0.35 if fresh else 0.0,
                "confidence_class": "LOW" if fresh else "BLOCKED",
                "legacy_unit_warning": (
                    "EXTERNAL_GEX_OMITS_SPOT_SQUARED_AND_ONE_PERCENT_SCALING"
                ),
                "dealer_position_observed": False,
                "authority": "CONTEXT_ONLY",
                "execution_authority": "NONE",
                "automatic_orders": 0,
                "warnings": [
                    "CURRENT_CHAIN_NOT_HISTORICAL_PIT",
                    "LEGACY_GEX_UNIT_NOT_COMPARABLE_TO_DOLLAR_GEX_1PCT",
                    "DEALER_POSITION_DIRECTION_IS_A_PROXY",
                ],
            }
        )
    return rows


def _normalize_chain(
    chain: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = chain.copy()
    frame.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    ]
    rename: dict[str, str] = {}
    for canonical, choices in ALIASES.items():
        existing = next(
            (candidate for candidate in choices if candidate in frame.columns),
            None,
        )
        if existing is not None:
            rename[existing] = canonical
    frame = frame.rename(columns=rename)
    required = {"expiration", "option_type", "strike", "open_interest"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GexInputError(f"option chain missing fields: {missing}")
    if "implied_volatility" not in frame:
        frame["implied_volatility"] = np.nan
    if "gamma" not in frame:
        frame["gamma"] = np.nan
    if "contract_multiplier" not in frame:
        frame["contract_multiplier"] = 100.0
    raw_rows = len(frame)
    frame["expiration"] = pd.to_datetime(
        frame["expiration"], utc=True, errors="coerce"
    )
    frame["option_type"] = (
        frame["option_type"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"c": "call", "p": "put", "calls": "call", "puts": "put"})
    )
    for column in (
        "strike",
        "open_interest",
        "implied_volatility",
        "gamma",
        "contract_multiplier",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["time_to_expiry_years"] = (
        (frame["expiration"] + pd.Timedelta(hours=21) - as_of)
        .dt.total_seconds()
        .div(365.25 * 86400.0)
    )
    valid_gamma = frame["gamma"].gt(0)
    valid_iv = frame["implied_volatility"].between(0.001, 5.0)
    frame = frame.loc[
        frame["option_type"].isin(["call", "put"])
        & frame["strike"].gt(0)
        & frame["open_interest"].gt(0)
        & frame["contract_multiplier"].gt(0)
        & frame["time_to_expiry_years"].gt(0)
        & (valid_gamma | valid_iv)
    ].copy()
    return frame, {
        "raw_rows": float(raw_rows),
        "valid_row_ratio": len(frame) / max(1, raw_rows),
    }


def _gamma_values(
    frame: pd.DataFrame,
    *,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[np.ndarray, np.ndarray]:
    provided = frame["gamma"].to_numpy(dtype=float)
    use_provided = np.isfinite(provided) & (provided > 0)
    sigma = frame["implied_volatility"].to_numpy(dtype=float)
    years = frame["time_to_expiry_years"].to_numpy(dtype=float)
    strike = frame["strike"].to_numpy(dtype=float)
    estimated = _black_scholes_gamma(
        spot=spot,
        strike=strike,
        sigma=sigma,
        years=years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    gamma = np.where(use_provided, provided, estimated)
    provenance = np.where(
        use_provided,
        "PROVIDER_OBSERVED",
        "BLACK_SCHOLES_ESTIMATED",
    )
    return gamma, provenance


def _black_scholes_gamma(
    *,
    spot: float,
    strike: np.ndarray,
    sigma: np.ndarray,
    years: np.ndarray,
    risk_free_rate: float,
    dividend_yield: float,
) -> np.ndarray:
    safe_sigma = np.where(
        np.isfinite(sigma) & (sigma > 0), sigma, np.nan
    )
    d1 = (
        np.log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * safe_sigma**2
        )
        * years
    ) / (safe_sigma * np.sqrt(years))
    normal_pdf = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return (
        np.exp(-dividend_yield * years)
        * normal_pdf
        / (spot * safe_sigma * np.sqrt(years))
    )


def _gex_scenario(
    *,
    strike: np.ndarray,
    years: np.ndarray,
    open_interest: np.ndarray,
    multiplier: np.ndarray,
    implied_volatility: np.ndarray,
    observed_gamma: np.ndarray,
    use_observed_gamma: np.ndarray,
    sign: np.ndarray,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    spots = np.linspace(spot * 0.80, spot * 1.20, 161)
    values: list[float] = []
    for scenario_spot in spots:
        estimated = _black_scholes_gamma(
            spot=float(scenario_spot),
            strike=strike,
            sigma=implied_volatility,
            years=years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        # Provider gamma is a point estimate at current spot. Scenario analysis
        # still needs a modelled surface, so estimated gamma is used off-spot.
        gamma = (
            np.where(use_observed_gamma, observed_gamma, estimated)
            if math.isclose(float(scenario_spot), spot, rel_tol=0, abs_tol=1e-9)
            else estimated
        )
        values.append(
            float(
                (
                    sign
                    * gamma
                    * open_interest
                    * multiplier
                    * float(scenario_spot) ** 2
                    * 0.01
                ).sum()
            )
        )
    return pd.DataFrame({"spot": spots, "net_gex_1pct": values})


def _nearest_zero_crossing(
    scenario: pd.DataFrame, spot: float
) -> float | None:
    x = scenario["spot"].to_numpy(dtype=float)
    y = scenario["net_gex_1pct"].to_numpy(dtype=float)
    signs = np.sign(y)
    indices = np.flatnonzero(signs[:-1] * signs[1:] <= 0)
    candidates: list[float] = []
    for index in indices:
        x1, x2 = float(x[index]), float(x[index + 1])
        y1, y2 = float(y[index]), float(y[index + 1])
        if math.isclose(y1, y2, abs_tol=1e-12):
            candidates.append((x1 + x2) / 2.0)
        else:
            candidates.append(x1 - y1 * (x2 - x1) / (y2 - y1))
    return min(candidates, key=lambda value: abs(value - spot)) if candidates else None


def _gex_regime(net_gex: float, gross_gex: float) -> str:
    if gross_gex <= 0 or abs(net_gex) / gross_gex < 0.02:
        return "NEUTRAL_GEX_PROXY"
    return "POSITIVE_GEX_PROXY" if net_gex > 0 else "NEGATIVE_GEX_PROXY"


def _sign_regime(value: Any) -> str:
    number = _finite_or_none(value)
    if number is None or math.isclose(number, 0.0, abs_tol=1e-12):
        return "NEUTRAL_GEX_PROXY"
    return "POSITIVE_GEX_PROXY" if number > 0 else "NEGATIVE_GEX_PROXY"


def _safe_share(numerator: Any, denominator: Any) -> float:
    den = float(denominator)
    return float(numerator) / den if den > 0 else 0.0


def _atr_distance(
    spot: float, level: float | None, atr_1d: float | None
) -> float | None:
    if level is None or atr_1d is None:
        return None
    atr = float(atr_1d)
    if not math.isfinite(atr) or atr <= 0:
        return None
    return round((spot - float(level)) / atr, 8)


def _confidence_class(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.50:
        return "MEDIUM"
    if value > 0:
        return "LOW"
    return "BLOCKED"


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


__all__ = [
    "GexInputError",
    "adapt_external_gex_snapshot",
    "calculate_gex_snapshot",
    "gex_schema",
]
