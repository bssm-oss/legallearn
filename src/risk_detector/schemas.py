from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ContractInput:
    contract_type: str = "jeonse"
    property_type: str = "villa"
    region: str = "수도권"
    deposit_million: float = 250.0
    monthly_rent_million: float = 0.0
    sale_price_million: float = 0.0
    estimated_market_price_million: float = 300.0
    mortgage_million: float = 0.0
    senior_claim_million: float = 0.0
    seizure: bool = False
    provisional_seizure: bool = False
    trust_registered: bool = False
    illegal_building: bool = False
    landlord_multiple_properties: bool = False
    landlord_prior_incidents: bool = False
    broker_unregistered: bool = False
    broker_advertising_issue: bool = False
    suspicious_special_clause: bool = False
    guarantee_insurance_available: bool = True
    fixed_date_ready: bool = True
    move_in_ready: bool = True
    broker_explained_rights: bool = True
    nearby_market_gap_percent: float = 0.0
    contract_period_months: int = 24
    special_clause_text: str = ""
    user_situation_text: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


RISK_LABELS = {
    0: "안전",
    1: "주의",
    2: "위험",
}

