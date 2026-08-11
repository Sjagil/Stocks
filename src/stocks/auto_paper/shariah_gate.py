from __future__ import annotations

from datetime import datetime

from stocks.auto_paper.contracts import AssetGroup, ShariahSnapshot


BLOCKED_SECURITY_TYPES = {"FUT", "OPT", "BOND", "CFD", "SWAP"}
BLOCKED_STRUCTURES = {"SHORT", "MARGIN", "LEVERAGED_ETF", "INVERSE_ETF", "SYNTHETIC_COMMODITY_ETP"}


def evaluate_shariah(
    *,
    security_type: str,
    asset_group: str,
    product_id: str,
    snapshot: ShariahSnapshot,
    product_allowlist: tuple[str, ...],
    decision_time: str,
) -> dict[str, object]:
    if security_type in BLOCKED_SECURITY_TYPES or snapshot.product_structure in BLOCKED_STRUCTURES:
        return _decision("SHARIAH_PRODUCT_STRUCTURE_BLOCKED")
    if snapshot.status == "SHARIAH_INELIGIBLE":
        return _decision("SHARIAH_INELIGIBLE")
    if snapshot.status == "SHARIAH_STATUS_STALE":
        return _decision("SHARIAH_STATUS_STALE")
    if snapshot.status == "SHARIAH_DATA_INCOMPLETE":
        return _decision("SHARIAH_DATA_INCOMPLETE")
    if snapshot.status == "SHARIAH_MANUAL_REVIEW_REQUIRED":
        return _decision("SHARIAH_MANUAL_REVIEW_REQUIRED")
    if snapshot.status != "SHARIAH_ELIGIBLE":
        return _decision("SHARIAH_DATA_INCOMPLETE")
    if not snapshot.methodology or not snapshot.methodology_version or not snapshot.financials_available_at:
        return _decision("SHARIAH_DATA_INCOMPLETE")
    if datetime.fromisoformat(decision_time) > datetime.fromisoformat(snapshot.expires_at):
        return _decision("SHARIAH_STATUS_STALE")
    if datetime.fromisoformat(snapshot.financials_available_at) > datetime.fromisoformat(decision_time):
        return _decision("SHARIAH_DATA_INCOMPLETE")
    if not (snapshot.business_activity_pass and snapshot.financial_ratio_pass and snapshot.non_permissible_income_pass):
        return _decision("SHARIAH_INELIGIBLE")

    if asset_group == AssetGroup.SHARIAH_STOCK and security_type == "STK":
        return _decision("SHARIAH_ELIGIBLE", eligible=True)
    if asset_group == AssetGroup.APPROVED_SHARIAH_EQUITY_ETF:
        if product_id not in product_allowlist:
            return _decision("SHARIAH_MANUAL_REVIEW_REQUIRED")
        if not snapshot.underlying_assets or not snapshot.shariah_certificate:
            return _decision("SHARIAH_DATA_INCOMPLETE")
        if snapshot.leverage or snapshot.short_exposure or snapshot.derivatives_exposure:
            return _decision("SHARIAH_PRODUCT_STRUCTURE_BLOCKED")
        if snapshot.interest_bearing_cash or snapshot.securities_lending or snapshot.currency_hedging:
            return _decision("SHARIAH_MANUAL_REVIEW_REQUIRED")
        return _decision("SHARIAH_ELIGIBLE", eligible=True)
    if asset_group == AssetGroup.APPROVED_PHYSICAL_COMMODITY_PRODUCT:
        if product_id not in product_allowlist:
            return _decision("SHARIAH_MANUAL_REVIEW_REQUIRED")
        if not snapshot.underlying_assets or not snapshot.shariah_certificate:
            return _decision("SHARIAH_DATA_INCOMPLETE")
        if not snapshot.physical_backing or snapshot.derivatives_exposure or snapshot.leverage:
            return _decision("SHARIAH_PRODUCT_STRUCTURE_BLOCKED")
        if snapshot.short_exposure or snapshot.interest_bearing_cash or snapshot.securities_lending or snapshot.currency_hedging:
            return _decision("SHARIAH_MANUAL_REVIEW_REQUIRED")
        return _decision("SHARIAH_ELIGIBLE", eligible=True)
    return _decision("SHARIAH_PRODUCT_STRUCTURE_BLOCKED")


def _decision(status: str, *, eligible: bool = False) -> dict[str, object]:
    return {
        "status": status,
        "eligible": eligible,
        "automatic_order_authority": "NONE",
        "manual_review_required": status == "SHARIAH_MANUAL_REVIEW_REQUIRED",
    }
