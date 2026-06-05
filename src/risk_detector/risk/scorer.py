from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from risk_detector.data.pipeline import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURE,
)
from risk_detector.features.signals import extract_text_signals
from risk_detector.paths import METADATA_PATH, MODEL_PATH
from risk_detector.schemas import ContractInput, RISK_LABELS


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "있음", "예", "위험"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def payload_to_contract(payload: dict[str, Any]) -> ContractInput:
    # 웹 폼과 JSON API는 문자열/숫자/불리언 입력이 섞여 들어오므로 한 번에 표준 스키마로 정규화한다.
    return ContractInput(
        contract_type=str(payload.get("contract_type", "jeonse")),
        property_type=str(payload.get("property_type", "villa")),
        region=str(payload.get("region", "수도권")),
        deposit_million=_as_float(payload.get("deposit_million"), 0.0),
        monthly_rent_million=_as_float(payload.get("monthly_rent_million"), 0.0),
        sale_price_million=_as_float(payload.get("sale_price_million"), 0.0),
        estimated_market_price_million=_as_float(payload.get("estimated_market_price_million"), 1.0),
        mortgage_million=_as_float(payload.get("mortgage_million"), 0.0),
        senior_claim_million=_as_float(payload.get("senior_claim_million"), 0.0),
        seizure=_as_bool(payload.get("seizure")),
        provisional_seizure=_as_bool(payload.get("provisional_seizure")),
        trust_registered=_as_bool(payload.get("trust_registered")),
        illegal_building=_as_bool(payload.get("illegal_building")),
        landlord_multiple_properties=_as_bool(payload.get("landlord_multiple_properties")),
        landlord_prior_incidents=_as_bool(payload.get("landlord_prior_incidents")),
        broker_unregistered=_as_bool(payload.get("broker_unregistered")),
        broker_advertising_issue=_as_bool(payload.get("broker_advertising_issue")),
        suspicious_special_clause=_as_bool(payload.get("suspicious_special_clause")),
        guarantee_insurance_available=_as_bool(payload.get("guarantee_insurance_available", True)),
        fixed_date_ready=_as_bool(payload.get("fixed_date_ready", True)),
        move_in_ready=_as_bool(payload.get("move_in_ready", True)),
        broker_explained_rights=_as_bool(payload.get("broker_explained_rights", True)),
        nearby_market_gap_percent=_as_float(payload.get("nearby_market_gap_percent"), 0.0),
        contract_period_months=_as_int(payload.get("contract_period_months"), 24),
        special_clause_text=str(payload.get("special_clause_text", "")),
        user_situation_text=str(payload.get("user_situation_text", "")),
    )


def infer_legal_category(contract: ContractInput) -> str:
    # 사용자가 입력한 특약/상황 텍스트를 학습 데이터의 법률 카테고리와 최대한 같은 축으로 맞춘다.
    combined = f"{contract.special_clause_text} {contract.user_situation_text}"
    if any(term in combined for term in ["사기", "기망", "편취", "고소"]):
        return "criminal_fraud"
    if any(term in combined for term in ["강박", "협박", "속였다", "허위"]):
        return "civil_fraud_duress"
    if any(term in combined for term in ["중개보조", "공인중개", "광고", "무등록"]):
        return "realtor_prohibited_acts"
    if contract.contract_type == "sale":
        return "civil_sale_effect"
    return "civil_lease_definition"


def contract_to_model_row(contract: ContractInput) -> dict[str, Any]:
    # 모델 학습 때 사용한 파생 피처와 같은 형태로 사용자 계약 입력을 변환한다.
    market = max(contract.estimated_market_price_million, 1.0)
    jeonse_ratio = contract.deposit_million / market
    debt_ratio = (contract.deposit_million + contract.mortgage_million + contract.senior_claim_million) / market
    has_registry_risk = contract.seizure or contract.provisional_seizure or contract.trust_registered
    has_contract_risk = (
        has_registry_risk
        or contract.illegal_building
        or contract.landlord_prior_incidents
        or contract.broker_unregistered
        or contract.broker_advertising_issue
        or contract.suspicious_special_clause
    )
    text = " ".join(
        [
            contract.contract_type,
            contract.property_type,
            contract.region,
            contract.special_clause_text,
            contract.user_situation_text,
            "신탁" if contract.trust_registered else "",
            "압류 가압류" if contract.seizure or contract.provisional_seizure else "",
            "위반건축물" if contract.illegal_building else "",
            "보증보험 불가" if not contract.guarantee_insurance_available else "보증보험 가능",
            "고전세가율" if contract.contract_type != "sale" and jeonse_ratio >= 0.85 else "",
            "높은 부채비율 선순위채권 위험" if debt_ratio >= 0.90 else "",
            "시세 괴리 허위계약 의심" if abs(contract.nearby_market_gap_percent) >= 18 else "",
            "임대인 보증사고" if contract.landlord_prior_incidents else "",
            "허위광고 중개보조원" if contract.broker_unregistered or contract.broker_advertising_issue else "",
            "권리침해 없음" if not has_contract_risk else "",
            "낮은 전세가율" if not has_contract_risk and contract.contract_type != "sale" and jeonse_ratio < 0.65 else "",
            "낮은 부채비율" if not has_contract_risk and debt_ratio < 0.72 else "",
            "정상 계약 보호요건 충족"
            if not has_contract_risk
            and contract.guarantee_insurance_available
            and contract.fixed_date_ready
            and contract.move_in_ready
            and contract.broker_explained_rights
            else "",
        ]
    )
    signals = extract_text_signals(text)
    row: dict[str, Any] = {
        **contract.to_dict(),
        "jeonse_ratio": round(jeonse_ratio, 4),
        "debt_ratio": round(debt_ratio, 4),
        "danger_term_count": signals.danger_term_count,
        "registry_term_count": signals.registry_term_count,
        "broker_term_count": signals.broker_term_count,
        "lease_term_count": signals.lease_term_count,
        "sale_term_count": signals.sale_term_count,
        "has_crime_signal": signals.has_crime_signal,
        "has_broker_signal": signals.has_broker_signal,
        "has_registry_signal": signals.has_registry_signal,
        "legal_category": infer_legal_category(contract),
        TEXT_FEATURE: text,
    }
    for col in NUMERIC_FEATURES:
        row[col] = _as_float(row.get(col), 0.0)
    for col in BOOLEAN_FEATURES:
        row[col] = int(_as_bool(row.get(col)))
    for col in CATEGORICAL_FEATURES:
        row[col] = str(row.get(col, "unknown"))
    return {col: row.get(col, "") for col in MODEL_FEATURES}


def rule_score_and_reasons(contract: ContractInput, model_row: dict[str, Any]) -> tuple[float, list[str], list[dict[str, Any]]]:
    # 법적으로 중요한 명시 신호는 모델이 낮게 봐도 최종 점수에 반영되도록 별도 규칙으로 보정한다.
    score = 12.0
    reasons: list[str] = []
    checks: list[dict[str, Any]] = []

    def add(points: float, label: str, severity: str, value: Any) -> None:
        nonlocal score
        score += points
        reasons.append(label)
        checks.append({"severity": severity, "points": points, "signal": label, "value": value})

    jeonse_ratio = float(model_row["jeonse_ratio"])
    debt_ratio = float(model_row["debt_ratio"])

    if contract.contract_type in {"jeonse", "monthly_rent"}:
        if jeonse_ratio >= 0.90:
            add(23, "전세가율이 90% 이상이라 보증금 회수 여력이 낮습니다.", "danger", f"{jeonse_ratio:.1%}")
        elif jeonse_ratio >= 0.75:
            add(18, "전세가율이 높은 편이라 주변 시세 재확인이 필요합니다.", "warning", f"{jeonse_ratio:.1%}")
        else:
            checks.append({"severity": "safe", "points": 0, "signal": "전세가율이 과도하지 않습니다.", "value": f"{jeonse_ratio:.1%}"})

    if debt_ratio >= 0.95:
        add(24, "보증금·근저당·선순위채권 합계가 시세에 근접하거나 초과합니다.", "danger", f"{debt_ratio:.1%}")
    elif debt_ratio >= 0.75:
        add(18, "부채비율이 높아 선순위 권리 확인이 필요합니다.", "warning", f"{debt_ratio:.1%}")

    if contract.mortgage_million > 0:
        add(min(12, contract.mortgage_million / max(contract.estimated_market_price_million, 1) * 20), "등기부 을구에 근저당 또는 담보성 채권이 있습니다.", "warning", contract.mortgage_million)
    if contract.senior_claim_million > 0:
        add(min(12, contract.senior_claim_million / max(contract.estimated_market_price_million, 1) * 24), "선순위 임차보증금 또는 선순위채권이 있습니다.", "warning", contract.senior_claim_million)
    if contract.seizure:
        add(24, "등기부 갑구에 압류가 있는 것으로 입력되었습니다.", "danger", True)
    if contract.provisional_seizure:
        add(21, "등기부 갑구에 가압류/가처분 등 권리침해 가능성이 있습니다.", "danger", True)
        if contract.contract_type == "sale":
            add(14, "매매 잔금 전 가압류/가처분은 소유권이전 이행 위험을 크게 높입니다.", "danger", True)
    if contract.trust_registered:
        add(22, "신탁등기가 있어 신탁원부와 임대 권한 확인이 필요합니다.", "danger", True)
    if contract.illegal_building:
        add(17, "건축물대장상 위반건축물 여부가 위험 신호입니다.", "danger", True)
    if contract.landlord_prior_incidents:
        add(22, "임대인 과거 보증사고/분쟁 이력이 입력되었습니다.", "danger", True)
    if contract.landlord_multiple_properties:
        add(7, "임대인의 다주택/동시 보증금 반환 부담 가능성이 있습니다.", "warning", True)
    if contract.broker_unregistered:
        add(18, "무등록 중개 또는 중개보조원 단독 진행 의심이 있습니다.", "danger", True)
    if contract.broker_advertising_issue:
        add(12, "광고 내용과 실제 권리관계가 다를 가능성이 있습니다.", "warning", True)
    if contract.suspicious_special_clause:
        add(13, "특약 조항에 임차인에게 불리하거나 권리 확인을 흐리는 내용이 있습니다.", "warning", True)
    if not contract.guarantee_insurance_available and contract.contract_type != "sale":
        add(20, "전세보증금반환보증 가입이 어렵거나 불가한 조건입니다.", "danger", False)
    if not contract.fixed_date_ready and contract.contract_type != "sale":
        add(11, "확정일자 확보가 불명확합니다.", "warning", False)
    if not contract.move_in_ready and contract.contract_type != "sale":
        add(11, "전입신고 또는 실제 점유 가능성이 불명확합니다.", "warning", False)
    if not contract.broker_explained_rights:
        add(9, "중개대상물 확인·설명이 충분하지 않은 것으로 입력되었습니다.", "warning", False)
    if abs(contract.nearby_market_gap_percent) >= 15:
        add(12, "주변 시세 대비 보증금/가격 괴리가 큽니다.", "warning", f"{contract.nearby_market_gap_percent:.1f}%")

    if not reasons:
        reasons.append("입력된 핵심 위험 신호가 낮은 편입니다. 그래도 등기부등본, 건축물대장, 보증보험 가능 여부는 계약 직전 재확인해야 합니다.")
    return min(score, 100.0), reasons, checks


def grade_from_score(score: float) -> str:
    if score >= 70:
        return "위험"
    if score >= 40:
        return "주의"
    return "안전"


class RiskScorer:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"학습된 모델이 없습니다: {model_path}")
        self.model = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    def score(self, payload: dict[str, Any] | ContractInput) -> dict[str, Any]:
        contract = payload if isinstance(payload, ContractInput) else payload_to_contract(payload)
        model_row = contract_to_model_row(contract)
        x = pd.DataFrame([model_row])
        predicted = int(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        probabilities = {RISK_LABELS[int(cls)]: round(float(proba[idx]), 4) for idx, cls in enumerate(classes)}
        risk_points = {
            0: 18,
            1: 55,
            2: 88,
        }
        model_score = sum(risk_points[int(cls)] * float(proba[idx]) for idx, cls in enumerate(classes))
        rule_score, reasons, checks = rule_score_and_reasons(contract, model_row)
        # 최종 점수는 모델 확률과 규칙 점수를 혼합한다. 데모 목적상 법적 위험 신호 누락을 줄이는 쪽에 초점을 둔다.
        blended = model_score * 0.58 + rule_score * 0.42
        final_score = max(blended, rule_score * 0.84)
        if contract.contract_type in {"jeonse", "monthly_rent"}:
            jeonse_ratio = float(model_row["jeonse_ratio"])
            debt_ratio = float(model_row["debt_ratio"])
            if jeonse_ratio >= 0.80 and debt_ratio >= 0.80:
                final_score = max(final_score, 40.0)
        if contract.trust_registered and not contract.guarantee_insurance_available and not contract.broker_explained_rights:
            final_score = max(final_score, 72.0)
        if (contract.seizure or contract.provisional_seizure) and not contract.guarantee_insurance_available:
            final_score = max(final_score, 78.0)
        if contract.landlord_prior_incidents and not contract.guarantee_insurance_available:
            final_score = max(final_score, 76.0)
        final_score = round(min(100.0, max(0.0, final_score)), 1)
        grade = grade_from_score(final_score)
        return {
            "risk_score": final_score,
            "risk_grade": grade,
            "model_predicted_grade": RISK_LABELS[predicted],
            "model_probabilities": probabilities,
            "model_score_component": round(model_score, 1),
            "rule_score_component": round(rule_score, 1),
            "reasons": reasons[:8],
            "checks": checks,
            "normalized_features": model_row,
            "legal_notice": "이 결과는 계약 전 위험 신호 탐지이며 수사기관 또는 법원의 법적 판단이 아닙니다.",
        }
