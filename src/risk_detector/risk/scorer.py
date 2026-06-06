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
    if any(term in combined for term in ["중개보조", "공인중개", "광고", "무등록", "폐업", "공제증서", "자격번호"]):
        return "realtor_prohibited_acts"
    if contract.contract_type == "sale":
        return "civil_sale_effect"
    return "civil_lease_definition"


def _text_risk_flags(text: str) -> dict[str, bool]:
    # 체크박스를 누락하고 자유 입력란에만 위험을 적는 실제 사용 패턴을 좁은 키워드 조합으로 보정한다.
    safe_resolution_terms = [
        "권리침해 없음",
        "등기부 깨끗",
        "갑구 을구 깨끗",
        "말소 완료",
        "말소등기 완료",
        "신탁원부 확인 완료",
        "수탁자 동의 완료",
        "보증보험 가능",
        "근저당 말소접수 완료",
        "대지권 등기 완료",
        "등기부 주소와 건축물대장 호수 일치",
        "영상통화 본인 확인",
        "신분증 원본 대조",
        "중개사 등록 확인 완료",
        "공제증서 확인 완료",
        "보증기관 조회 완료",
        "계약서 실제 보증금 일치",
    ]
    has_safe_resolution = any(term in text for term in safe_resolution_terms)
    registry_risk = any(term in text for term in ["압류 있는데", "압류가", "압류 표시", "가압류", "가처분", "곧 풀린다"]) and not has_safe_resolution
    trust_risk = any(term in text for term in ["신탁원부 없어도", "신탁 표시", "신탁등기", "수탁자 동의 나중", "임대 권한 불명"]) and not any(
        term in text for term in ["신탁원부 확인 완료", "수탁자 동의 완료"]
    )
    no_guarantee_risk = any(term in text for term in ["보증보험 불가", "보증보험 안", "보증보험 안됨", "보증보험 가입이 안", "보증보험 어렵"])
    account_risk = any(
        term in text
        for term in ["계좌 명의", "다른 사람 계좌", "임대인 이름이 아닙", "대표 개인계좌", "개인계좌로 보내", "현금 요구"]
    ) and not any(term in text for term in ["임대인 계좌 일치", "소유자 계좌 확인", "계좌 확인 완료"])
    building_safe = any(term in text for term in ["건축물대장 확인 완료", "건축물대장 정상", "위반건축물 아님", "불법증축 아님", "무허가 아님"])
    illegal_building_risk = any(term in text for term in ["불법증축", "쪼개기", "무허가", "용도위반", "건축물대장 못", "위반건축물"]) and not building_safe
    pressure_risk = any(term in text for term in ["계약금 먼저", "오늘 계약금", "빨리 입금", "등기부는 나중", "서류는 나중"]) and not any(
        term in text for term in ["계약금 보류", "등기부 확인 완료", "서류 원본 확인"]
    )
    sublease_safe = any(term in text for term in ["전대차가 아니라", "임대인 본인과 직접 계약", "임대인 동의서 확인 완료"])
    sublease_risk = any(term in text for term in ["전대차", "기존 세입자", "임대인 동의서 없음", "임대인 동의서는 없"]) and not sublease_safe
    lease_registration_risk = "임차권등기명령" in text and not any(term in text for term in ["임차권등기명령 말소 완료", "말소 완료", "말소등기 완료"])
    unregistered_safe = any(term in text for term in ["소유권보존등기 완료", "사용승인 완료", "등기부 확인 완료"])
    unregistered_risk = any(term in text for term in ["미등기", "사용승인 전", "등기부등본이 없"]) and not unregistered_safe
    reservation_deposit_risk = "가계약금" in text and any(term in text for term in ["환불불가", "계약서 보기 전", "먼저 송금"]) and not any(
        term in text for term in ["가계약금 환불 가능", "계약서 확인 완료", "환불 가능"]
    )
    priority_safe = any(
        term in text
        for term in [
            "전입세대열람 확인 완료",
            "확정일자 부여현황 확인 완료",
            "선순위 보증금 총액 확인 완료",
            "전체 선순위 보증금 확인 완료",
        ]
    )
    priority_disclosure_risk = (
        any(term in text for term in ["전입세대열람", "전입세대 열람", "확정일자 부여현황"])
        and any(term in text for term in ["못 보여", "안 보여", "거부", "집주인만", "확인 어렵"])
        and not priority_safe
    )
    senior_deposit_unknown_risk = (
        any(term in text for term in ["선순위 임차보증금 총액", "선순위 보증금 총액", "방마다 보증금", "전체 선순위 보증금"])
        and any(term in text for term in ["모른", "안 알려", "확인 어렵", "개인정보"])
        and not priority_safe
    )
    auction_safe = any(term in text for term in ["경매 공매 없음", "경매 없음", "공매 없음", "취하 완료", "말소 완료"])
    auction_risk = any(
        term in text for term in ["경매개시결정", "임의경매개시결정", "강제경매개시결정", "경매개시", "공매 예고", "공매 통지"]
    ) and not auction_safe
    ownership_safe = any(term in text for term in ["소유자 변경 없음", "매도인 임대인 일치", "승계 확인 완료", "소유자 확인 완료"])
    ownership_change_risk = any(
        term in text for term in ["소유자가 바뀔", "소유자 변경", "매매와 전세를 동시에", "전세를 동시에", "동시진행", "갭투자 승계"]
    ) and not ownership_safe
    same_day_loan_safe = any(term in text for term in ["잔금 당일 대출 없음", "전입 후 대출 금지 특약", "당일 근저당 없음"])
    same_day_loan_risk = (
        (
            any(term in text for term in ["잔금 당일", "전입신고 전에", "전입 전에"])
            and any(term in text for term in ["대출", "근저당"])
        )
        or "대출을 먼저 실행" in text
    ) and not same_day_loan_safe
    land_right_safe = any(term in text for term in ["대지권 등기 완료", "대지권 확인 완료", "토지 지분 확인 완료"])
    land_right_risk = (
        any(term in text for term in ["대지권 미등기", "대지권 없음"])
        or ("토지 지분" in text and any(term in text for term in ["나중", "미정리", "불명", "없"]))
    ) and not land_right_safe
    unit_mismatch_safe = any(term in text for term in ["주소 호수 일치", "등기부 주소와 건축물대장 호수 일치", "계약서 호수 확인 완료"])
    unit_mismatch_risk = (
        any(term in text for term in ["호수 불일치", "주소 불일치"])
        or (
            any(term in text for term in ["등기부 주소", "건축물대장 호수", "실제 보는 방", "계약서 호수"])
            and any(term in text for term in ["다르", "불일치", "맞지 않", "틀리"])
        )
    ) and not unit_mismatch_safe
    free_residence_safe = any(term in text for term in ["무상거주확인서 없음", "무상거주 확인서 없음"])
    free_residence_risk = (
        any(term in text for term in ["무상거주확인서", "무상거주 확인서"])
        and any(term in text for term in ["서명", "실제 보증금", "따로", "대출 심사", "작성"])
        and not free_residence_safe
    )
    jeonse_right_safe = any(term in text for term in ["전세권 설정 가능", "전세권 설정 완료", "전세권 설정 협의 완료"])
    jeonse_right_refusal_risk = (
        "전세권 설정" in text
        and any(term in text for term in ["거부", "절대 안", "안 된", "불가", "해줄 수 없"])
        and not jeonse_right_safe
    )
    corporate_authority_safe = any(
        term in text for term in ["법인등기부등본 확인 완료", "사용인감계 확인 완료", "인감증명 확인 완료"]
    )
    corporate_authority_risk = (
        (
            "법인" in text
            and any(term in text for term in ["법인등기부등본", "사용인감계", "인감증명", "직원 명함"])
            and any(term in text for term in ["없이", "미확인", "명함만", "없"])
        )
        or "직원 명함만" in text
        or "법인등기부등본 사용인감계 인감증명 미확인" in text
    ) and not corporate_authority_safe
    mortgage_cancellation_safe = any(term in text for term in ["근저당 말소접수 완료", "말소등기 완료", "근저당 말소 완료"])
    mortgage_cancellation_risk = (
        "근저당" in text
        and any(term in text for term in ["말소접수 전", "영수증만", "잔금으로 갚", "먼저 입금", "먼저 송금"])
        and not mortgage_cancellation_safe
    )
    remote_identity_safe = any(
        term in text for term in ["영상통화 본인 확인", "신분증 원본 대조", "위임장 원본", "인감증명 원본", "본인 확인 완료"]
    )
    remote_identity_risk = (
        any(term in text for term in ["해외 체류", "비대면", "대리인", "대리인이 계약"])
        and any(term in text for term in ["신분증 사진만", "영상통화 거부", "원본 못", "사진만", "카톡으로만"])
        and not remote_identity_safe
    )
    broker_license_safe = any(term in text for term in ["중개사 등록 확인 완료", "공제증서 확인 완료", "등록증 확인 완료", "공제증서 원본 확인"])
    broker_license_risk = (
        any(term in text for term in ["폐업 상태", "폐업", "등록증 조회 안됨", "공제증서 나중", "공제증서 만료", "자격번호 없음", "등록번호 없음", "무자격"])
        and not broker_license_safe
    )
    guarantee_document_safe = any(term in text for term in ["보증기관 조회 완료", "보증서 원본 확인", "HUG 조회 완료", "보증보험 가능"])
    guarantee_document_risk = (
        any(term in text for term in ["보증서 캡처", "PDF 캡처", "HUG 보증서", "보증서가 이미 발급", "보증기관 조회"])
        and any(term in text for term in ["하지 말", "하지말", "캡처만", "조회는", "확인하지 말"])
        and not guarantee_document_safe
    )
    down_contract_safe = any(term in text for term in ["계약서 실제 보증금 일치", "차액 현금 없음", "실제 지급액 일치"])
    down_contract_risk = (
        any(term in text for term in ["다운계약", "보증금을 낮게", "계약서에는 보증금을 낮게", "차액은 현금", "실제 보증금 차액", "세금이 줄어"])
        and not down_contract_safe
    )
    return {
        "registry_text_risk": registry_risk,
        "trust_text_risk": trust_risk,
        "no_guarantee_text_risk": no_guarantee_risk,
        "account_text_risk": account_risk,
        "illegal_building_text_risk": illegal_building_risk,
        "pressure_text_risk": pressure_risk,
        "sublease_text_risk": sublease_risk,
        "lease_registration_text_risk": lease_registration_risk,
        "unregistered_text_risk": unregistered_risk,
        "reservation_deposit_text_risk": reservation_deposit_risk,
        "priority_disclosure_text_risk": priority_disclosure_risk,
        "senior_deposit_unknown_text_risk": senior_deposit_unknown_risk,
        "auction_text_risk": auction_risk,
        "ownership_change_text_risk": ownership_change_risk,
        "same_day_loan_text_risk": same_day_loan_risk,
        "land_right_text_risk": land_right_risk,
        "unit_mismatch_text_risk": unit_mismatch_risk,
        "free_residence_text_risk": free_residence_risk,
        "jeonse_right_refusal_text_risk": jeonse_right_refusal_risk,
        "corporate_authority_text_risk": corporate_authority_risk,
        "mortgage_cancellation_text_risk": mortgage_cancellation_risk,
        "remote_identity_text_risk": remote_identity_risk,
        "broker_license_text_risk": broker_license_risk,
        "guarantee_document_text_risk": guarantee_document_risk,
        "down_contract_text_risk": down_contract_risk,
    }


def contract_to_model_row(contract: ContractInput) -> dict[str, Any]:
    # 모델 학습 때 사용한 파생 피처와 같은 형태로 사용자 계약 입력을 변환한다.
    market = max(contract.estimated_market_price_million, 1.0)
    jeonse_ratio = contract.deposit_million / market
    debt_ratio = (contract.deposit_million + contract.mortgage_million + contract.senior_claim_million) / market
    user_text = f"{contract.special_clause_text} {contract.user_situation_text}"
    text_flags = _text_risk_flags(user_text)
    has_text_risk = any(text_flags.values())
    has_registry_risk = contract.seizure or contract.provisional_seizure or contract.trust_registered
    has_contract_risk = (
        has_registry_risk
        or contract.illegal_building
        or contract.landlord_prior_incidents
        or contract.broker_unregistered
        or contract.broker_advertising_issue
        or contract.suspicious_special_clause
        or has_text_risk
    )
    if not contract.guarantee_insurance_available or text_flags["no_guarantee_text_risk"]:
        guarantee_text = "보증보험 불가"
    elif has_text_risk:
        guarantee_text = ""
    else:
        guarantee_text = "보증보험 가능"
    text = " ".join(
        [
            contract.contract_type,
            contract.property_type,
            contract.region,
            contract.special_clause_text,
            contract.user_situation_text,
            "신탁" if contract.trust_registered or text_flags["trust_text_risk"] else "",
            "압류 가압류" if contract.seizure or contract.provisional_seizure or text_flags["registry_text_risk"] else "",
            "위반건축물" if contract.illegal_building or text_flags["illegal_building_text_risk"] else "",
            guarantee_text,
            "명의 불일치 계약금 계좌" if text_flags["account_text_risk"] else "",
            "계약 먼저 등기부는 나중" if text_flags["pressure_text_risk"] else "",
            "전대차 임대인 동의서 없음" if text_flags["sublease_text_risk"] else "",
            "임차권등기명령 말소 미확인" if text_flags["lease_registration_text_risk"] else "",
            "미등기 사용승인 전" if text_flags["unregistered_text_risk"] else "",
            "가계약금 환불불가 계약서 보기 전" if text_flags["reservation_deposit_text_risk"] else "",
            "전입세대열람 확정일자 부여현황 미제공" if text_flags["priority_disclosure_text_risk"] else "",
            "선순위 임차보증금 총액 불명" if text_flags["senior_deposit_unknown_text_risk"] else "",
            "경매개시결정 공매 예고" if text_flags["auction_text_risk"] else "",
            "소유자 변경 매매 전세 동시진행" if text_flags["ownership_change_text_risk"] else "",
            "잔금 당일 대출 전입 전 근저당" if text_flags["same_day_loan_text_risk"] else "",
            "대지권 미등기 토지 지분 미정리" if text_flags["land_right_text_risk"] else "",
            "등기부 주소 건축물대장 호수 계약서 호수 불일치" if text_flags["unit_mismatch_text_risk"] else "",
            "무상거주확인서 실제 보증금 따로" if text_flags["free_residence_text_risk"] else "",
            "전세권 설정 거부" if text_flags["jeonse_right_refusal_text_risk"] else "",
            "법인등기부등본 사용인감계 인감증명 미확인" if text_flags["corporate_authority_text_risk"] else "",
            "근저당 말소접수 전 영수증만 먼저 입금" if text_flags["mortgage_cancellation_text_risk"] else "",
            "비대면 계약 신분증 사진만 영상통화 거부" if text_flags["remote_identity_text_risk"] else "",
            "폐업 중개사무소 공제증서 나중 자격번호 없음" if text_flags["broker_license_text_risk"] else "",
            "HUG 보증서 PDF 캡처 보증기관 조회 거부" if text_flags["guarantee_document_text_risk"] else "",
            "다운계약 보증금 낮게 차액 현금" if text_flags["down_contract_text_risk"] else "",
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
        "safety_term_count": signals.safety_term_count,
        "critical_term_count": signals.critical_term_count,
        "has_crime_signal": signals.has_crime_signal,
        "has_broker_signal": signals.has_broker_signal,
        "has_registry_signal": signals.has_registry_signal,
        "has_safety_signal": signals.has_safety_signal,
        "has_critical_signal": signals.has_critical_signal,
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
    combined_text = f"{contract.special_clause_text} {contract.user_situation_text}"

    def add(points: float, label: str, severity: str, value: Any) -> None:
        nonlocal score
        score += points
        reasons.append(label)
        checks.append({"severity": severity, "points": points, "signal": label, "value": value})

    def has_any(terms: list[str]) -> bool:
        return any(term in combined_text for term in terms)

    text_flags = _text_risk_flags(combined_text)
    identity_mismatch = has_any(["명의 불일치", "신분증 불일치", "위임장 미확인", "인감증명 미확인"]) or (
        "불일치" in combined_text and any(term in combined_text for term in ["명의", "신분증", "소유자"])
    )
    proxy_docs_deferred = any(term in combined_text for term in ["나중", "계약 후", "추후"]) and any(
        term in combined_text for term in ["위임장", "인감증명", "대리권"]
    )

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
    if text_flags["registry_text_risk"]:
        add(24, "자유 입력 문장에 압류·가압류 등 등기부 권리침해 정황이 있습니다.", "danger", True)
    if text_flags["trust_text_risk"]:
        add(22, "자유 입력 문장에 신탁원부 또는 수탁자 동의 미확인 정황이 있습니다.", "danger", True)
    if text_flags["no_guarantee_text_risk"] and contract.contract_type != "sale":
        add(16, "자유 입력 문장에 보증보험 가입 불가 또는 거절 정황이 있습니다.", "danger", True)
    if text_flags["account_text_risk"]:
        add(23, "계약금 또는 보증금 입금 계좌가 임대인/소유자 명의와 다를 가능성이 있습니다.", "danger", True)
    if text_flags["illegal_building_text_risk"]:
        add(18, "자유 입력 문장에 불법증축, 쪼개기, 무허가 등 건축물대장 확인 위험이 있습니다.", "danger", True)
    if text_flags["pressure_text_risk"]:
        add(14, "등기부·서류 확인 전 계약금 선입금 또는 급박한 계약 압박 정황이 있습니다.", "warning", True)
    if text_flags["sublease_text_risk"]:
        add(20, "임대인 동의 없는 전대차 또는 기존 세입자와의 계약 정황이 있습니다.", "danger", True)
    if text_flags["lease_registration_text_risk"]:
        add(20, "등기부에 임차권등기명령 말소 미확인 정황이 있습니다.", "danger", True)
    if text_flags["unregistered_text_risk"]:
        add(22, "미등기 또는 사용승인 전 신축으로 권리관계 확인이 어렵습니다.", "danger", True)
    if text_flags["reservation_deposit_text_risk"]:
        add(16, "계약서 확인 전 가계약금 환불불가 또는 선송금 압박이 있습니다.", "warning", True)
    if text_flags["priority_disclosure_text_risk"]:
        add(18, "전입세대열람 또는 확정일자 부여현황 확인을 거부하는 정황이 있습니다.", "warning", True)
    if text_flags["senior_deposit_unknown_text_risk"]:
        add(22, "선순위 임차보증금 총액을 확인할 수 없어 보증금 회수 순위가 불명확합니다.", "danger", True)
    if text_flags["auction_text_risk"]:
        add(26, "경매개시결정 또는 공매 예고 정황이 있어 계약 전 권리관계 확인이 필요합니다.", "danger", True)
    if text_flags["ownership_change_text_risk"]:
        add(20, "계약 전후 소유자 변경 또는 매매·전세 동시진행 정황이 있습니다.", "danger", True)
    if text_flags["same_day_loan_text_risk"]:
        add(22, "잔금 당일 대출 선행 또는 전입 전 근저당 설정 위험이 있습니다.", "danger", True)
    if text_flags["land_right_text_risk"]:
        add(20, "대지권 미등기 또는 토지 지분 미정리로 권리관계 확인이 어렵습니다.", "danger", True)
    if text_flags["unit_mismatch_text_risk"]:
        add(20, "등기부·건축물대장·계약서의 주소 또는 호수가 서로 다를 가능성이 있습니다.", "danger", True)
    if text_flags["free_residence_text_risk"]:
        add(24, "무상거주확인서 작성과 실제 보증금 분리 정황은 허위 계약·대출 위험 신호입니다.", "danger", True)
    if text_flags["jeonse_right_refusal_text_risk"]:
        add(16, "전세권 설정을 과도하게 거부하는 정황이 있어 보증금 보호 수단을 재검토해야 합니다.", "warning", True)
    if text_flags["corporate_authority_text_risk"]:
        add(22, "법인 임대인의 권한 서류 또는 사용인감 확인이 부족합니다.", "danger", True)
    if text_flags["mortgage_cancellation_text_risk"]:
        add(22, "근저당 말소접수 전 영수증만 제시하고 입금을 요구하는 정황이 있습니다.", "danger", True)
    if text_flags["remote_identity_text_risk"]:
        add(22, "비대면 임대인 또는 대리계약에서 신분증 원본·영상통화 확인이 부족합니다.", "danger", True)
    if text_flags["broker_license_text_risk"]:
        add(20, "중개사무소 폐업, 자격번호 누락, 공제증서 미확인 등 중개 자격 위험이 있습니다.", "danger", True)
    if text_flags["guarantee_document_text_risk"]:
        add(20, "보증서 캡처만 제시하고 보증기관 조회를 막는 정황이 있습니다.", "danger", True)
    if text_flags["down_contract_text_risk"]:
        add(24, "계약서 금액과 실제 보증금 차액을 분리하는 다운계약·허위계약 위험이 있습니다.", "danger", True)
    if identity_mismatch or proxy_docs_deferred:
        add(22, "임대인 신원 또는 대리권 확인에 중대한 불일치가 있습니다.", "danger", True)
    if has_any(["전입신고 지연", "전입 전 근저당", "당일 근저당", "대항력 포기", "잔금 후 담보대출"]):
        add(20, "전입·확정일자 전 권리공백 또는 당일 담보 설정 위험이 있습니다.", "danger", True)
    if has_any(["국세 체납", "지방세 체납", "당해세", "체납 압류"]):
        add(18, "임대인의 세금 체납 또는 당해세 우선 변제 위험이 의심됩니다.", "danger", True)
    if has_any(["이중계약", "중복계약", "선순위보증금 숨김", "다중 임차인"]):
        add(22, "이중계약 또는 선순위보증금 은폐 가능성이 있습니다.", "danger", True)

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
        text = f"{contract.special_clause_text} {contract.user_situation_text}"
        text_flags = _text_risk_flags(text)
        if text_flags["registry_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["trust_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["no_guarantee_text_risk"] and contract.contract_type != "sale":
            final_score = max(final_score, 64.0)
            if float(model_row["jeonse_ratio"]) >= 0.80 or abs(contract.nearby_market_gap_percent) >= 15:
                final_score = max(final_score, 70.0)
        if text_flags["account_text_risk"]:
            final_score = max(final_score, 74.0)
        if text_flags["illegal_building_text_risk"]:
            final_score = max(final_score, 70.0)
        if text_flags["pressure_text_risk"]:
            final_score = max(final_score, 64.0)
        if text_flags["sublease_text_risk"]:
            final_score = max(final_score, 70.0)
        if text_flags["lease_registration_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["unregistered_text_risk"]:
            final_score = max(final_score, 74.0)
        if text_flags["reservation_deposit_text_risk"]:
            final_score = max(final_score, 64.0)
        if text_flags["priority_disclosure_text_risk"]:
            final_score = max(final_score, 66.0)
        if text_flags["senior_deposit_unknown_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["auction_text_risk"]:
            final_score = max(final_score, 78.0)
        if text_flags["ownership_change_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["same_day_loan_text_risk"]:
            final_score = max(final_score, 76.0)
        if text_flags["land_right_text_risk"]:
            final_score = max(final_score, 74.0)
        if text_flags["unit_mismatch_text_risk"]:
            final_score = max(final_score, 70.0)
        if text_flags["free_residence_text_risk"]:
            final_score = max(final_score, 76.0)
        if text_flags["jeonse_right_refusal_text_risk"]:
            final_score = max(final_score, 64.0)
        if text_flags["corporate_authority_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["mortgage_cancellation_text_risk"]:
            final_score = max(final_score, 74.0)
        if text_flags["remote_identity_text_risk"]:
            final_score = max(final_score, 74.0)
        if text_flags["broker_license_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["guarantee_document_text_risk"]:
            final_score = max(final_score, 72.0)
        if text_flags["down_contract_text_risk"]:
            final_score = max(final_score, 76.0)
        identity_mismatch = any(term in text for term in ["명의 불일치", "신분증 불일치", "위임장 미확인", "인감증명 미확인"]) or (
            "불일치" in text and any(term in text for term in ["명의", "신분증", "소유자"])
        )
        proxy_docs_deferred = any(term in text for term in ["나중", "계약 후", "추후"]) and any(
            term in text for term in ["위임장", "인감증명", "대리권"]
        )
        if identity_mismatch or proxy_docs_deferred:
            final_score = max(final_score, 74.0)
        if any(term in text for term in ["전입신고 지연", "전입 전 근저당", "당일 근저당", "대항력 포기", "잔금 후 담보대출"]):
            final_score = max(final_score, 76.0)
        if any(term in text for term in ["국세 체납", "지방세 체납", "당해세", "체납 압류"]):
            final_score = max(final_score, 72.0)
        if any(term in text for term in ["이중계약", "중복계약", "선순위보증금 숨김", "다중 임차인"]):
            final_score = max(final_score, 78.0)
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
