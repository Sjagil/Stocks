from __future__ import annotations

from stocks.alpha.data_contracts import InstrumentType, ShariahScreen, ShariahStatus


ALLOWED_INSTRUMENT_TYPES = {
    InstrumentType.STOCK,
    InstrumentType.SHARIAH_EQUITY_ETF,
    InstrumentType.APPROVED_PHYSICAL_COMMODITY_PRODUCT,
}

BLOCKED_INSTRUMENT_TYPES = {
    InstrumentType.BOND,
    InstrumentType.FUTURE,
    InstrumentType.OPTION,
    InstrumentType.SWAP,
    InstrumentType.CFD,
    InstrumentType.SHORT,
    InstrumentType.LEVERAGED_ETF,
    InstrumentType.INVERSE_ETF,
    InstrumentType.SYNTHETIC_ETF,
}


def validate_shariah_screen(screen: ShariahScreen) -> dict[str, object]:
    reasons = list(screen.rejection_reasons)
    if screen.instrument_type in BLOCKED_INSTRUMENT_TYPES:
        reasons.append(f"BLOCKED_ASSET_CLASS_{screen.instrument_type.value}")
    if screen.instrument_type not in ALLOWED_INSTRUMENT_TYPES:
        reasons.append("ASSET_CLASS_NOT_IN_ALLOWED_UNIVERSE")
    if screen.compliance_status != ShariahStatus.ELIGIBLE:
        reasons.append(screen.compliance_status.value)
    if screen.has_derivatives:
        reasons.append(ShariahStatus.DERIVATIVE_EXPOSURE_BLOCKED.value)
    if screen.has_futures:
        reasons.append(ShariahStatus.FUTURES_EXPOSURE_BLOCKED.value)
    if screen.has_leverage:
        reasons.append("LEVERAGE_ZERO_FAILED")
    if screen.has_short_exposure:
        reasons.append("SHORT_EXPOSURE_ZERO_FAILED")
    if screen.has_interest_bearing_cash:
        reasons.append(ShariahStatus.INTEREST_EXPOSURE_BLOCKED.value)
    status = "GO" if not reasons else "NO_GO"
    return {
        "status": status,
        "instrument_id": screen.instrument_id,
        "compliance_status": screen.compliance_status.value,
        "instrument_type": screen.instrument_type.value,
        "rejection_reasons": sorted(set(reasons)),
    }
