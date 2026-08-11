from __future__ import annotations


def parse_market_rule_ids(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        raise ValueError("marketRuleIds string is blank")

    rule_ids: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("marketRuleIds contains an empty value")
        if not part.isdigit():
            raise ValueError(f"marketRuleIds contains a non-integer value: {part}")
        rule_id = int(part)
        if rule_id <= 0:
            raise ValueError(f"marketRuleIds contains a non-positive value: {part}")
        rule_ids.append(rule_id)

    return tuple(rule_ids)


def validate_market_rule_ids(value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError("market_rule_ids is required")
    try:
        parse_market_rule_ids(value)
    except ValueError as exc:
        raise ValueError(f"market_rule_ids must be comma-separated positive integers: {exc}") from exc
