from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from risk_detector.features.signals import extract_text_signals
from risk_detector.paths import (
    DATA_DIR,
    DERIVED_CONTRACTS_CSV,
    EXTERNAL_CASE_REFERENCES_CSV,
    MANUAL_SCENARIOS_CSV,
    PUBLIC_RISK_INDICATORS_CSV,
    REFERENCE_SOURCES_MD,
    SOURCE_CASES_CSV,
)


NUMERIC_FEATURES = [
    "deposit_million",
    "monthly_rent_million",
    "sale_price_million",
    "estimated_market_price_million",
    "mortgage_million",
    "senior_claim_million",
    "jeonse_ratio",
    "debt_ratio",
    "nearby_market_gap_percent",
    "contract_period_months",
    "danger_term_count",
    "registry_term_count",
    "broker_term_count",
    "lease_term_count",
    "sale_term_count",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "property_type",
    "region",
    "legal_category",
]

BOOLEAN_FEATURES = [
    "seizure",
    "provisional_seizure",
    "trust_registered",
    "illegal_building",
    "landlord_multiple_properties",
    "landlord_prior_incidents",
    "broker_unregistered",
    "broker_advertising_issue",
    "suspicious_special_clause",
    "guarantee_insurance_available",
    "fixed_date_ready",
    "move_in_ready",
    "broker_explained_rights",
    "has_crime_signal",
    "has_broker_signal",
    "has_registry_signal",
]

TEXT_FEATURE = "combined_text"
TARGET_COLUMN = "risk_label"
SCORE_COLUMN = "heuristic_risk_score"

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES + [TEXT_FEATURE]


HIGH_RISK_CATEGORIES = {
    "criminal_fraud",
    "aggravated_economic_crime",
    "embezzlement_breach_of_trust",
}

CAUTION_CATEGORIES = {
    "civil_fraud_duress",
    "realtor_prohibited_acts",
}


PUBLIC_RISK_INDICATORS = [
    {
        "indicator_id": "PUB_SAFE_RT_LOW_RATIO",
        "source_type": "official_market",
        "source_name": "국토교통부 실거래가 공개시스템",
        "source_url": "https://rt.molit.go.kr/pt/info/info.do?mobileAt=v",
        "risk_theme": "실거래가 기준 저전세가율 정상 계약",
        "risk_label": 0,
        "recommended_feature": "jeonse_ratio, nearby_market_gap_percent",
        "evidence_summary": "부동산 거래신고 자료를 주변 시세 검증 기준으로 사용한다.",
    },
    {
        "indicator_id": "PUB_CAUTION_RT_GAP",
        "source_type": "official_market",
        "source_name": "국토교통부 아파트 매매 실거래가 자료",
        "source_url": "https://www.data.go.kr/data/15126469/openapi.do",
        "risk_theme": "주변 실거래가 대비 보증금 또는 매매가 괴리",
        "risk_label": 1,
        "recommended_feature": "nearby_market_gap_percent",
        "evidence_summary": "실거래 신고자료와 입력 가격의 괴리를 주의 신호로 사용한다.",
    },
    {
        "indicator_id": "PUB_DANGER_HUG_LTV90",
        "source_type": "official_guarantee",
        "source_name": "HUG 전세보증금반환보증 상품개요",
        "source_url": "https://m.khug.or.kr/hug/web/ig/dr/igdr000001.jsp?tabMenu=Y",
        "risk_theme": "보증금과 선순위채권 합산액이 주택가격 90% 기준을 초과",
        "risk_label": 2,
        "recommended_feature": "debt_ratio, guarantee_insurance_available",
        "evidence_summary": "보증보험 가능 여부와 선순위채권 합산 비율을 핵심 위험 신호로 사용한다.",
    },
    {
        "indicator_id": "PUB_DANGER_HUG_INCIDENT",
        "source_type": "official_guarantee",
        "source_name": "HUG 전세 및 임대보증 공공데이터",
        "source_url": "https://khug.or.kr/houstar/web/p03/01/p030105.jsp?articleId=34712&currentPage=1&mode=S",
        "risk_theme": "전세보증 사고·대위변제 이력 기반 지역/임대인 위험",
        "risk_label": 2,
        "recommended_feature": "landlord_prior_incidents, landlord_multiple_properties",
        "evidence_summary": "보증사고 공개자료는 향후 임대인/지역 위험 보강 후보로 사용한다.",
    },
    {
        "indicator_id": "PUB_DANGER_REGISTRY_SEIZURE",
        "source_type": "official_registry",
        "source_name": "대법원 인터넷등기소",
        "source_url": "https://www.iros.go.kr",
        "risk_theme": "등기부 갑구 압류·가압류·가처분 권리침해",
        "risk_label": 2,
        "recommended_feature": "seizure, provisional_seizure",
        "evidence_summary": "등기부 권리침해 표시는 계약 전 중대한 위험 신호다.",
    },
    {
        "indicator_id": "PUB_DANGER_TRUST",
        "source_type": "official_registry",
        "source_name": "대법원 인터넷등기소",
        "source_url": "https://www.iros.go.kr",
        "risk_theme": "신탁등기와 임대 권한 확인 누락",
        "risk_label": 2,
        "recommended_feature": "trust_registered, broker_explained_rights",
        "evidence_summary": "신탁원부와 임대 권한 확인이 누락되면 보증금 회수 위험이 커진다.",
    },
    {
        "indicator_id": "PUB_DANGER_BUILDING",
        "source_type": "official_building",
        "source_name": "국토교통부 건축HUB 건축물대장정보 서비스",
        "source_url": "https://www.data.go.kr/data/15134735/openapi.do",
        "risk_theme": "건축물대장상 위반건축물",
        "risk_label": 2,
        "recommended_feature": "illegal_building",
        "evidence_summary": "건축물대장 속성은 위반건축물 여부 확인 후보 데이터다.",
    },
    {
        "indicator_id": "PUB_CAUTION_BROKER_AD",
        "source_type": "official_law",
        "source_name": "공인중개사법 제18조의2, 제33조",
        "source_url": "https://www.law.go.kr/법령/공인중개사법",
        "risk_theme": "허위·과장 광고 또는 무등록 중개 의심",
        "risk_label": 1,
        "recommended_feature": "broker_unregistered, broker_advertising_issue",
        "evidence_summary": "중개대상물 부당 표시·광고와 금지행위를 중개 위험 신호로 사용한다.",
    },
    {
        "indicator_id": "PUB_SAFE_PROTECTION",
        "source_type": "official_law",
        "source_name": "주택임대차보호법",
        "source_url": "https://www.law.go.kr/법령/주택임대차보호법",
        "risk_theme": "전입·확정일자·보증보험 가능 정상 보호 조건",
        "risk_label": 0,
        "recommended_feature": "fixed_date_ready, move_in_ready, guarantee_insurance_available",
        "evidence_summary": "보호 요건이 모두 가능한 경우 정상 계약 기준 사례로 사용한다.",
    },
    {
        "indicator_id": "PUB_DANGER_FALSE_CONTRACT",
        "source_type": "case_news",
        "source_name": "대법원 전세보증금 부풀린 계약서 보도",
        "source_url": "https://www.hankyung.com/article/202506220823i",
        "risk_theme": "보증금 부풀림·허위 계약서 기반 대출/보증 위험",
        "risk_label": 2,
        "recommended_feature": "nearby_market_gap_percent, suspicious_special_clause",
        "evidence_summary": "실제 지급액과 계약서 보증금이 크게 다르면 허위 계약 위험으로 반영한다.",
    },
]


EXTERNAL_CASE_REFERENCES = [
    {
        "reference_id": "EXT_CASE_001",
        "source_name": "국가법령정보센터 판례",
        "source_url": "https://www.law.go.kr/LSW/precInfoP.do?precSeq=97060",
        "case_number": "82도2428",
        "risk_theme": "전세보증금 채권 양도와 사기죄 성립 경계",
        "risk_label_hint": 1,
        "evidence_summary": "전세보증금 관련 채권 이전·처분 구조를 경계 사례로 참조한다.",
    },
    {
        "reference_id": "EXT_CASE_002",
        "source_name": "한국경제 대법원 보도",
        "source_url": "https://www.hankyung.com/article/202506220823i",
        "case_number": "대법원 2025.5.29. 선고 보증채무금 관련 사건",
        "risk_theme": "전세보증금 부풀림과 허위 전세계약",
        "risk_label_hint": 2,
        "evidence_summary": "계약서상 보증금과 실제 지급 보증금 차이를 허위 계약 위험으로 반영한다.",
    },
    {
        "reference_id": "EXT_CASE_003",
        "source_name": "한국경제 부동산 법률 칼럼 판결 소개",
        "source_url": "https://www.hankyung.com/article/202305015715Q",
        "case_number": "1심 대출사기 의심 사건",
        "risk_theme": "매매가보다 높은 전세금, 감정평가·대출사기 의심",
        "risk_label_hint": 2,
        "evidence_summary": "매매가보다 전세금이 높은 구조를 고위험 학습 예시로 참조한다.",
    },
]


def load_case_rows(path: Path = SOURCE_CASES_CSV) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def risk_label_from_score(score: float) -> int:
    # 파생 데이터 라벨은 법적 확정 판정이 아니라 계약 전 위험도 학습용 등급이다.
    if score >= 70:
        return 2
    if score >= 40:
        return 1
    return 0


def _case_base_score(row: dict[str, str], signals) -> float:
    # 판례의 법률 카테고리, 사건 유형, 주문, 위험 키워드를 점수화해 계약 위험 라벨의 기준값을 만든다.
    category = row.get("legal_category", "")
    case_type = row.get("case_type", "")
    disposition = row.get("disposition", "")
    score = 28.0

    if category in HIGH_RISK_CATEGORIES:
        score += 36
    elif category in CAUTION_CATEGORIES:
        score += 24
    elif category in {"civil_lease_definition", "civil_sale_effect"}:
        score += 13

    if case_type == "형사":
        score += 14
    if any(term in disposition for term in ["유죄", "징역", "벌금"]):
        score += 8
    if any(term in disposition for term in ["무죄", "청구기각"]):
        score -= 7

    score += min(signals.danger_term_count * 3.5, 20)
    score += min(signals.registry_term_count * 1.2, 10)
    score += min(signals.broker_term_count * 1.5, 8)
    return _clip(score, 5, 98)


def _pick_contract_type(row: dict[str, str], rng: random.Random) -> str:
    text = " ".join([row.get("case_name", ""), row.get("keywords", ""), row.get("summary", "")])
    if any(term in text for term in ["전세", "임대차", "보증금", "임차"]):
        return rng.choices(["jeonse", "monthly_rent"], weights=[0.78, 0.22], k=1)[0]
    if any(term in text for term in ["매매", "소유권이전", "분양", "매매대금"]):
        return "sale"
    return rng.choice(["jeonse", "sale", "monthly_rent"])


def _pick_property_type(row: dict[str, str], rng: random.Random) -> str:
    text = " ".join([row.get("case_name", ""), row.get("keywords", ""), row.get("summary", "")])
    if "오피스텔" in text:
        return "officetel"
    if "다가구" in text or "다세대" in text or "빌라" in text:
        return "villa"
    if "상가" in text or "상업" in text:
        return "commercial"
    if "아파트" in text:
        return "apartment"
    return rng.choices(["villa", "apartment", "officetel", "multi_family"], weights=[0.36, 0.30, 0.18, 0.16], k=1)[0]


def _contract_features_from_case(row: dict[str, str], rng: random.Random, variant_id: int = 0) -> dict[str, object]:
    # 판례 한 건에서 여러 계약 조건 변형을 만들어 Bagging이 다양한 위험 조합을 보도록 한다.
    combined_text = " ".join(
        [
            row.get("case_name", ""),
            row.get("summary", ""),
            row.get("holding", ""),
            row.get("statutes", ""),
            row.get("keywords", ""),
            row.get("legal_category_ko", ""),
        ]
    )
    signals = extract_text_signals(combined_text)
    base_score = _case_base_score(row, signals)
    label = risk_label_from_score(base_score)
    contract_type = _pick_contract_type(row, rng)
    property_type = _pick_property_type(row, rng)
    market = round(rng.uniform(160, 950), 1)

    if label == 2:
        jeonse_ratio = rng.uniform(0.88, 1.12)
        mortgage_ratio = rng.uniform(0.25, 0.66)
        senior_ratio = rng.uniform(0.08, 0.35)
    elif label == 1:
        jeonse_ratio = rng.uniform(0.68, 0.92)
        mortgage_ratio = rng.uniform(0.05, 0.36)
        senior_ratio = rng.uniform(0.00, 0.20)
    else:
        jeonse_ratio = rng.uniform(0.35, 0.72)
        mortgage_ratio = rng.uniform(0.00, 0.18)
        senior_ratio = rng.uniform(0.00, 0.08)

    if contract_type == "sale":
        sale_price = round(market * rng.uniform(0.86, 1.08), 1)
        deposit = round(sale_price * rng.uniform(0.05, 0.18), 1)
        monthly_rent = 0.0
    elif contract_type == "monthly_rent":
        sale_price = 0.0
        deposit = round(market * rng.uniform(0.08, 0.32), 1)
        monthly_rent = round(rng.uniform(0.35, 2.4), 2)
    else:
        sale_price = 0.0
        deposit = round(market * jeonse_ratio, 1)
        monthly_rent = 0.0

    mortgage = round(market * mortgage_ratio, 1)
    senior_claim = round(market * senior_ratio, 1)
    debt_ratio = round((deposit + mortgage + senior_claim) / max(market, 1), 4)
    price_gap = round((deposit + sale_price - market) / max(market, 1) * 100, 2)

    seizure = label == 2 and rng.random() < 0.24
    provisional_seizure = label == 2 and rng.random() < 0.22
    trust = (label == 2 and rng.random() < 0.18) or ("신탁" in combined_text and rng.random() < 0.55)
    illegal_building = label >= 1 and rng.random() < (0.10 if label == 1 else 0.20)
    landlord_incident = label == 2 and rng.random() < 0.30
    broker_unregistered = ("공인중개사법" in combined_text or "중개" in combined_text) and label >= 1 and rng.random() < 0.22
    broker_ad_issue = ("표시" in combined_text or "광고" in combined_text or "중개" in combined_text) and label >= 1 and rng.random() < 0.30
    suspicious_clause = label >= 1 and rng.random() < (0.18 if label == 1 else 0.38)

    guarantee_available = not (
        debt_ratio > 0.90
        or seizure
        or provisional_seizure
        or trust
        or illegal_building
        or contract_type == "sale"
    )

    return {
        "source": "case_derived",
        "source_case_number": row.get("case_number", ""),
        "source_case_name": row.get("case_name", ""),
        "derivation_variant_id": variant_id,
        "contract_type": contract_type,
        "property_type": property_type,
        "region": rng.choice(["수도권", "비수도권", "광역시", "지방중소도시"]),
        "deposit_million": deposit,
        "monthly_rent_million": monthly_rent,
        "sale_price_million": sale_price,
        "estimated_market_price_million": round(market, 1),
        "mortgage_million": mortgage,
        "senior_claim_million": senior_claim,
        "jeonse_ratio": round(deposit / max(market, 1), 4),
        "debt_ratio": debt_ratio,
        "nearby_market_gap_percent": price_gap,
        "contract_period_months": rng.choice([12, 24, 24, 24, 36]),
        "seizure": seizure,
        "provisional_seizure": provisional_seizure,
        "trust_registered": trust,
        "illegal_building": illegal_building,
        "landlord_multiple_properties": label >= 1 and rng.random() < (0.22 if label == 1 else 0.48),
        "landlord_prior_incidents": landlord_incident,
        "broker_unregistered": broker_unregistered,
        "broker_advertising_issue": broker_ad_issue,
        "suspicious_special_clause": suspicious_clause,
        "guarantee_insurance_available": guarantee_available,
        "fixed_date_ready": not (label == 2 and rng.random() < 0.18),
        "move_in_ready": not (label == 2 and rng.random() < 0.12),
        "broker_explained_rights": not (label >= 1 and rng.random() < (0.25 if label == 1 else 0.52)),
        "danger_term_count": signals.danger_term_count,
        "registry_term_count": signals.registry_term_count,
        "broker_term_count": signals.broker_term_count,
        "lease_term_count": signals.lease_term_count,
        "sale_term_count": signals.sale_term_count,
        "has_crime_signal": signals.has_crime_signal,
        "has_broker_signal": signals.has_broker_signal,
        "has_registry_signal": signals.has_registry_signal,
        "legal_category": row.get("legal_category", "unknown"),
        "combined_text": combined_text[:2800],
        SCORE_COLUMN: round(base_score, 2),
        TARGET_COLUMN: label,
    }


def _safe_synthetic_examples(count: int, rng: random.Random) -> list[dict[str, object]]:
    # 실제 판례 데이터는 위험 사건 중심이라 정상 계약 기준 예시를 별도로 보강한다.
    rows: list[dict[str, object]] = []
    for idx in range(count):
        contract_type = rng.choice(["jeonse", "monthly_rent", "sale"])
        market = round(rng.uniform(180, 850), 1)
        if contract_type == "sale":
            sale_price = round(market * rng.uniform(0.92, 1.03), 1)
            deposit = round(sale_price * rng.uniform(0.05, 0.12), 1)
            monthly = 0.0
            legal_category = "civil_sale_effect"
        elif contract_type == "monthly_rent":
            sale_price = 0.0
            deposit = round(market * rng.uniform(0.05, 0.24), 1)
            monthly = round(rng.uniform(0.4, 1.6), 2)
            legal_category = "civil_lease_definition"
        else:
            sale_price = 0.0
            deposit = round(market * rng.uniform(0.38, 0.68), 1)
            monthly = 0.0
            legal_category = "civil_lease_definition"
        mortgage = round(market * rng.uniform(0.0, 0.12), 1)
        senior = round(market * rng.uniform(0.0, 0.05), 1)
        combined = (
            "정상 계약 예시. 등기부 권리침해 없음, 선순위채권 낮음, "
            "공인중개사 확인설명 완료, 전입신고와 확정일자 예정, 보증보험 가능."
        )
        rows.append(
            {
                "source": "synthetic_safe_reference",
                "source_case_number": f"SAFE_SYN_{idx+1:04d}",
                "source_case_name": "정상 계약 기준 예시",
                "derivation_variant_id": 0,
                "contract_type": contract_type,
                "property_type": rng.choice(["apartment", "villa", "officetel", "multi_family"]),
                "region": rng.choice(["수도권", "비수도권", "광역시", "지방중소도시"]),
                "deposit_million": deposit,
                "monthly_rent_million": monthly,
                "sale_price_million": sale_price,
                "estimated_market_price_million": market,
                "mortgage_million": mortgage,
                "senior_claim_million": senior,
                "jeonse_ratio": round(deposit / max(market, 1), 4),
                "debt_ratio": round((deposit + mortgage + senior) / max(market, 1), 4),
                "nearby_market_gap_percent": round((deposit + sale_price - market) / max(market, 1) * 100, 2),
                "contract_period_months": rng.choice([24, 24, 36]),
                "seizure": False,
                "provisional_seizure": False,
                "trust_registered": False,
                "illegal_building": False,
                "landlord_multiple_properties": rng.random() < 0.08,
                "landlord_prior_incidents": False,
                "broker_unregistered": False,
                "broker_advertising_issue": False,
                "suspicious_special_clause": False,
                "guarantee_insurance_available": contract_type != "sale",
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
                "danger_term_count": 0,
                "registry_term_count": 1,
                "broker_term_count": 1,
                "lease_term_count": 1 if contract_type != "sale" else 0,
                "sale_term_count": 1 if contract_type == "sale" else 0,
                "has_crime_signal": False,
                "has_broker_signal": True,
                "has_registry_signal": True,
                "legal_category": legal_category,
                "combined_text": combined,
                SCORE_COLUMN: round(rng.uniform(10, 34), 2),
                TARGET_COLUMN: 0,
            }
        )
    return rows


def _hard_boundary_examples(count: int, rng: random.Random) -> list[dict[str, object]]:
    # 전세가율, 신탁, 가압류처럼 실제 계약에서 헷갈리는 경계 사례를 고정 템플릿으로 만든다.
    rows: list[dict[str, object]] = []
    templates = [
        {
            "label": 2,
            "source": "synthetic_hard_danger",
            "text": "가압류 신탁 보증보험 불가 고전세가율 근저당 선순위채권 설명 누락",
            "contract_type": "jeonse",
            "property_type": "villa",
            "jeonse_range": (0.91, 1.08),
            "mortgage_range": (0.26, 0.58),
            "senior_range": (0.08, 0.30),
            "flags": {
                "provisional_seizure": True,
                "trust_registered": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
                "suspicious_special_clause": True,
            },
        },
        {
            "label": 2,
            "source": "synthetic_hard_danger",
            "text": "위반건축물 다가구 선순위보증금 불명 보증보험 불가 근저당",
            "contract_type": "monthly_rent",
            "property_type": "multi_family",
            "jeonse_range": (0.28, 0.58),
            "mortgage_range": (0.32, 0.64),
            "senior_range": (0.20, 0.46),
            "flags": {
                "illegal_building": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 1,
            "source": "synthetic_hard_caution",
            "text": "전세가율 높음 근저당 일부 있음 주변 시세 괴리 확인 필요",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "jeonse_range": (0.72, 0.88),
            "mortgage_range": (0.08, 0.28),
            "senior_range": (0.00, 0.16),
            "flags": {"guarantee_insurance_available": True},
        },
        {
            "label": 1,
            "source": "synthetic_hard_caution",
            "text": "중개보조원 설명 허위광고 의심 권리관계 재확인 필요",
            "contract_type": "jeonse",
            "property_type": "villa",
            "jeonse_range": (0.62, 0.82),
            "mortgage_range": (0.02, 0.18),
            "senior_range": (0.00, 0.10),
            "flags": {
                "broker_unregistered": True,
                "broker_advertising_issue": True,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 0,
            "source": "synthetic_hard_safe",
            "text": "권리침해 없음 낮은 전세가율 보증보험 가능 확정일자 전입 가능",
            "contract_type": "jeonse",
            "property_type": "apartment",
            "jeonse_range": (0.36, 0.62),
            "mortgage_range": (0.00, 0.08),
            "senior_range": (0.00, 0.03),
            "flags": {
                "guarantee_insurance_available": True,
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 0,
            "source": "synthetic_hard_safe",
            "text": "매매 잔금 전 권리침해 없음 말소특약 명확 소유권이전 준비 완료",
            "contract_type": "sale",
            "property_type": "apartment",
            "jeonse_range": (0.04, 0.12),
            "mortgage_range": (0.00, 0.16),
            "senior_range": (0.00, 0.03),
            "flags": {
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
    ]
    for idx in range(count):
        template = templates[idx % len(templates)]
        label = int(template["label"])
        market = round(rng.uniform(120, 1200), 1)
        contract_type = str(template["contract_type"])
        jeonse_ratio = rng.uniform(*template["jeonse_range"])
        mortgage_ratio = rng.uniform(*template["mortgage_range"])
        senior_ratio = rng.uniform(*template["senior_range"])
        if contract_type == "sale":
            sale_price = round(market * rng.uniform(0.88, 1.10), 1)
            deposit = round(sale_price * rng.uniform(0.04, 0.16), 1)
            monthly_rent = 0.0
            legal_category = "civil_sale_effect"
        elif contract_type == "monthly_rent":
            sale_price = 0.0
            deposit = round(market * jeonse_ratio, 1)
            monthly_rent = round(rng.uniform(0.35, 2.8), 2)
            legal_category = "civil_lease_definition"
        else:
            sale_price = 0.0
            deposit = round(market * jeonse_ratio, 1)
            monthly_rent = 0.0
            legal_category = "civil_lease_definition"
        mortgage = round(market * mortgage_ratio, 1)
        senior = round(market * senior_ratio, 1)
        flags = {
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": label >= 1 and rng.random() < 0.45,
            "landlord_prior_incidents": label == 2 and rng.random() < 0.32,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": label >= 1 and rng.random() < 0.45,
            "guarantee_insurance_available": contract_type != "sale" and label == 0,
            "fixed_date_ready": label != 2,
            "move_in_ready": label != 2,
            "broker_explained_rights": label == 0,
        }
        flags.update(template["flags"])
        signals = extract_text_signals(str(template["text"]))
        score = {0: rng.uniform(8, 34), 1: rng.uniform(43, 66), 2: rng.uniform(76, 98)}[label]
        rows.append(
            {
                "source": template["source"],
                "source_case_number": f"HARD_{idx + 1:05d}",
                "source_case_name": "서비스형 경계/스트레스 학습 예시",
                "derivation_variant_id": 0,
                "contract_type": contract_type,
                "property_type": template["property_type"],
                "region": rng.choice(["수도권", "비수도권", "광역시", "지방중소도시"]),
                "deposit_million": deposit,
                "monthly_rent_million": monthly_rent,
                "sale_price_million": sale_price,
                "estimated_market_price_million": market,
                "mortgage_million": mortgage,
                "senior_claim_million": senior,
                "jeonse_ratio": round(deposit / max(market, 1), 4),
                "debt_ratio": round((deposit + mortgage + senior) / max(market, 1), 4),
                "nearby_market_gap_percent": round(rng.uniform(-8, 24) if label else rng.uniform(-10, 7), 2),
                "contract_period_months": rng.choice([0, 12, 24, 36]) if contract_type == "sale" else rng.choice([12, 24, 36]),
                **flags,
                "danger_term_count": signals.danger_term_count,
                "registry_term_count": signals.registry_term_count,
                "broker_term_count": signals.broker_term_count,
                "lease_term_count": signals.lease_term_count,
                "sale_term_count": signals.sale_term_count,
                "has_crime_signal": signals.has_crime_signal,
                "has_broker_signal": signals.has_broker_signal,
                "has_registry_signal": signals.has_registry_signal,
                "legal_category": legal_category,
                "combined_text": str(template["text"]),
                SCORE_COLUMN: round(score, 2),
                TARGET_COLUMN: label,
            }
        )
    return rows


def _public_indicator_examples(count: int, rng: random.Random) -> list[dict[str, object]]:
    # 공식 공개자료의 판단 기준을 모델이 학습할 수 있도록 오프라인 계약 예시로 변환한다.
    rows: list[dict[str, object]] = []
    templates = [
        {
            "label": 0,
            "source": "public_indicator_safe",
            "indicator_id": "PUB_SAFE_RT_LOW_RATIO",
            "text": "실거래가 대조 완료 낮은 전세가율 권리침해 없음 보증보험 가능 확정일자 전입 가능",
            "contract_type": "jeonse",
            "property_type": "apartment",
            "jeonse_range": (0.42, 0.62),
            "mortgage_range": (0.00, 0.06),
            "senior_range": (0.00, 0.03),
            "gap_range": (-6, 4),
            "legal_category": "civil_lease_definition",
            "flags": {
                "guarantee_insurance_available": True,
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 0,
            "source": "public_indicator_safe",
            "indicator_id": "PUB_SAFE_PROTECTION",
            "text": "등기부 말소사항 포함 확인 권리침해 없음 선순위채권 없음 정상 계약",
            "contract_type": "monthly_rent",
            "property_type": "apartment",
            "jeonse_range": (0.08, 0.28),
            "mortgage_range": (0.00, 0.08),
            "senior_range": (0.00, 0.04),
            "gap_range": (-8, 5),
            "legal_category": "civil_lease_definition",
            "flags": {
                "guarantee_insurance_available": True,
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 1,
            "source": "public_indicator_caution",
            "indicator_id": "PUB_CAUTION_RT_GAP",
            "text": "주변 실거래가 대비 보증금 괴리 큼 시세 재확인 필요",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "jeonse_range": (0.68, 0.84),
            "mortgage_range": (0.03, 0.16),
            "senior_range": (0.00, 0.10),
            "gap_range": (12, 27),
            "legal_category": "civil_lease_definition",
            "flags": {"guarantee_insurance_available": True},
        },
        {
            "label": 2,
            "source": "public_indicator_danger",
            "indicator_id": "PUB_DANGER_HUG_LTV90",
            "text": "보증금 선순위채권 합산액 주택가격 90퍼센트 초과 보증보험 불가",
            "contract_type": "jeonse",
            "property_type": "villa",
            "jeonse_range": (0.88, 1.08),
            "mortgage_range": (0.08, 0.38),
            "senior_range": (0.08, 0.28),
            "gap_range": (8, 28),
            "legal_category": "civil_lease_definition",
            "flags": {
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 2,
            "source": "public_indicator_danger",
            "indicator_id": "PUB_DANGER_REGISTRY_SEIZURE",
            "text": "등기부 갑구 압류 가압류 가처분 권리침해 잔금 전 재확인 필요",
            "contract_type": "sale",
            "property_type": "apartment",
            "jeonse_range": (0.04, 0.14),
            "mortgage_range": (0.18, 0.42),
            "senior_range": (0.00, 0.08),
            "gap_range": (-4, 12),
            "legal_category": "civil_sale_effect",
            "flags": {
                "seizure": True,
                "provisional_seizure": True,
                "fixed_date_ready": False,
                "move_in_ready": False,
                "suspicious_special_clause": True,
            },
        },
        {
            "label": 2,
            "source": "public_indicator_danger",
            "indicator_id": "PUB_DANGER_TRUST",
            "text": "신탁등기 신탁원부 임대 권한 확인 누락 보증금 반환 책임 불명",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "jeonse_range": (0.72, 0.94),
            "mortgage_range": (0.00, 0.22),
            "senior_range": (0.04, 0.22),
            "gap_range": (4, 18),
            "legal_category": "civil_lease_definition",
            "flags": {
                "trust_registered": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
                "suspicious_special_clause": True,
            },
        },
        {
            "label": 2,
            "source": "public_indicator_danger",
            "indicator_id": "PUB_DANGER_BUILDING",
            "text": "건축물대장 위반건축물 다가구 선순위보증금 확인 불가",
            "contract_type": "monthly_rent",
            "property_type": "multi_family",
            "jeonse_range": (0.25, 0.52),
            "mortgage_range": (0.20, 0.52),
            "senior_range": (0.18, 0.42),
            "gap_range": (2, 18),
            "legal_category": "civil_lease_definition",
            "flags": {
                "illegal_building": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 1,
            "source": "public_indicator_caution",
            "indicator_id": "PUB_CAUTION_BROKER_AD",
            "text": "공인중개사법 부당 표시광고 허위매물 중개보조원 단독 설명 의심",
            "contract_type": "jeonse",
            "property_type": "villa",
            "jeonse_range": (0.58, 0.78),
            "mortgage_range": (0.02, 0.20),
            "senior_range": (0.00, 0.12),
            "gap_range": (5, 16),
            "legal_category": "realtor_prohibited_acts",
            "flags": {
                "broker_unregistered": True,
                "broker_advertising_issue": True,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 2,
            "source": "public_indicator_danger",
            "indicator_id": "PUB_DANGER_HUG_INCIDENT",
            "text": "전세보증 사고 대위변제 임대인 다주택 동시 보증금 반환 부담",
            "contract_type": "jeonse",
            "property_type": "villa",
            "jeonse_range": (0.78, 1.02),
            "mortgage_range": (0.12, 0.46),
            "senior_range": (0.04, 0.24),
            "gap_range": (6, 24),
            "legal_category": "criminal_fraud",
            "flags": {
                "landlord_multiple_properties": True,
                "landlord_prior_incidents": True,
                "guarantee_insurance_available": False,
            },
        },
        {
            "label": 2,
            "source": "public_indicator_danger",
            "indicator_id": "PUB_DANGER_FALSE_CONTRACT",
            "text": "전세보증금 부풀림 허위 전세계약 보증 대출 사기 의심",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "jeonse_range": (0.95, 1.18),
            "mortgage_range": (0.00, 0.18),
            "senior_range": (0.00, 0.16),
            "gap_range": (20, 42),
            "legal_category": "criminal_fraud",
            "flags": {
                "suspicious_special_clause": True,
                "broker_advertising_issue": True,
                "guarantee_insurance_available": False,
            },
        },
    ]

    for idx in range(count):
        template = templates[idx % len(templates)]
        label = int(template["label"])
        market = round(rng.uniform(140, 1300), 1)
        contract_type = str(template["contract_type"])
        jeonse_ratio = rng.uniform(*template["jeonse_range"])
        mortgage_ratio = rng.uniform(*template["mortgage_range"])
        senior_ratio = rng.uniform(*template["senior_range"])
        if contract_type == "sale":
            sale_price = round(market * rng.uniform(0.92, 1.18), 1)
            deposit = round(sale_price * rng.uniform(0.04, 0.14), 1)
            monthly_rent = 0.0
        elif contract_type == "monthly_rent":
            sale_price = 0.0
            deposit = round(market * jeonse_ratio, 1)
            monthly_rent = round(rng.uniform(0.35, 2.6), 2)
        else:
            sale_price = 0.0
            deposit = round(market * jeonse_ratio, 1)
            monthly_rent = 0.0
        mortgage = round(market * mortgage_ratio, 1)
        senior = round(market * senior_ratio, 1)
        flags = {
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": label >= 1 and rng.random() < 0.28,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": label >= 1 and rng.random() < 0.34,
            "guarantee_insurance_available": contract_type != "sale" and label == 0,
            "fixed_date_ready": label != 2,
            "move_in_ready": label != 2,
            "broker_explained_rights": label == 0,
        }
        flags.update(template["flags"])
        signals = extract_text_signals(str(template["text"]))
        score = {0: rng.uniform(6, 31), 1: rng.uniform(42, 66), 2: rng.uniform(74, 99)}[label]
        rows.append(
            {
                "source": template["source"],
                "source_case_number": f"{template['indicator_id']}_{idx + 1:05d}",
                "source_case_name": f"공개자료 지표 기반 예시: {template['indicator_id']}",
                "derivation_variant_id": idx,
                "contract_type": contract_type,
                "property_type": template["property_type"],
                "region": rng.choice(["수도권", "비수도권", "광역시", "지방중소도시"]),
                "deposit_million": deposit,
                "monthly_rent_million": monthly_rent,
                "sale_price_million": sale_price,
                "estimated_market_price_million": market,
                "mortgage_million": mortgage,
                "senior_claim_million": senior,
                "jeonse_ratio": round(deposit / max(market, 1), 4),
                "debt_ratio": round((deposit + mortgage + senior) / max(market, 1), 4),
                "nearby_market_gap_percent": round(rng.uniform(*template["gap_range"]), 2),
                "contract_period_months": rng.choice([0, 12, 24, 36]) if contract_type == "sale" else rng.choice([12, 24, 24, 36]),
                **flags,
                "danger_term_count": signals.danger_term_count,
                "registry_term_count": signals.registry_term_count,
                "broker_term_count": signals.broker_term_count,
                "lease_term_count": signals.lease_term_count,
                "sale_term_count": signals.sale_term_count,
                "has_crime_signal": signals.has_crime_signal,
                "has_broker_signal": signals.has_broker_signal,
                "has_registry_signal": signals.has_registry_signal,
                "legal_category": template["legal_category"],
                "combined_text": str(template["text"]),
                SCORE_COLUMN: round(score, 2),
                TARGET_COLUMN: label,
            }
        )
    return rows


def _counterfactual_stress_examples(count: int, rng: random.Random) -> list[dict[str, object]]:
    # 추가학습용 반례 세트다. 전세가율/부채비율 하나만 보고 판단하지 않도록 법적 신호와 안전 신호를 교차시킨다.
    rows: list[dict[str, object]] = []
    templates = [
        {
            "label": 0,
            "source": "synthetic_counterfactual_safe",
            "text": "근저당 일부 있으나 낮은 전세가율 선순위채권 없음 보증보험 가능 확정일자 전입 가능 공인중개사 확인설명 완료",
            "contract_type": "jeonse",
            "property_type": "apartment",
            "legal_category": "civil_lease_definition",
            "jeonse_range": (0.42, 0.62),
            "mortgage_range": (0.05, 0.16),
            "senior_range": (0.00, 0.04),
            "gap_range": (-8, 4),
            "flags": {
                "guarantee_insurance_available": True,
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 0,
            "source": "synthetic_counterfactual_safe",
            "text": "다가구 월세 전체 선순위보증금 목록 확인 근저당 낮음 위반건축물 아님 보증금 소액 보호요건 충족",
            "contract_type": "monthly_rent",
            "property_type": "multi_family",
            "legal_category": "civil_lease_definition",
            "jeonse_range": (0.06, 0.22),
            "mortgage_range": (0.02, 0.12),
            "senior_range": (0.02, 0.10),
            "gap_range": (-5, 5),
            "flags": {
                "guarantee_insurance_available": True,
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 0,
            "source": "synthetic_counterfactual_safe",
            "text": "매매 잔금 전 말소특약 명확 권리침해 없음 소유권이전 준비 완료 가격 시세 범위",
            "contract_type": "sale",
            "property_type": "apartment",
            "legal_category": "civil_sale_effect",
            "jeonse_range": (0.04, 0.12),
            "mortgage_range": (0.04, 0.18),
            "senior_range": (0.00, 0.03),
            "gap_range": (-4, 6),
            "flags": {
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 1,
            "source": "synthetic_counterfactual_caution",
            "text": "전세가율 높지만 압류 신탁 위반건축물 없음 보증보험 가능 주변 시세와 권리관계 재확인 필요",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "legal_category": "civil_lease_definition",
            "jeonse_range": (0.76, 0.88),
            "mortgage_range": (0.00, 0.12),
            "senior_range": (0.00, 0.06),
            "gap_range": (4, 15),
            "flags": {
                "guarantee_insurance_available": True,
                "fixed_date_ready": True,
                "move_in_ready": True,
                "broker_explained_rights": True,
            },
        },
        {
            "label": 1,
            "source": "synthetic_counterfactual_caution",
            "text": "매매 가격이 주변 실거래가보다 낮아 하자 권리관계 확인 필요 압류는 없으나 특약 재검토",
            "contract_type": "sale",
            "property_type": "villa",
            "legal_category": "civil_sale_effect",
            "jeonse_range": (0.04, 0.13),
            "mortgage_range": (0.00, 0.18),
            "senior_range": (0.00, 0.04),
            "gap_range": (-22, -12),
            "flags": {
                "broker_explained_rights": True,
            },
        },
        {
            "label": 1,
            "source": "synthetic_counterfactual_caution",
            "text": "월세 보증금은 낮지만 중개보조원 설명 비중이 높고 광고 내용 일부 불일치 확인 필요",
            "contract_type": "monthly_rent",
            "property_type": "villa",
            "legal_category": "realtor_prohibited_acts",
            "jeonse_range": (0.08, 0.28),
            "mortgage_range": (0.02, 0.18),
            "senior_range": (0.00, 0.08),
            "gap_range": (2, 12),
            "flags": {
                "broker_unregistered": True,
                "broker_advertising_issue": True,
                "broker_explained_rights": False,
                "guarantee_insurance_available": True,
            },
        },
        {
            "label": 2,
            "source": "synthetic_counterfactual_danger",
            "text": "전세가율은 낮아 보이나 등기부 갑구 압류 가압류 권리침해로 보증금 회수 위험",
            "contract_type": "jeonse",
            "property_type": "apartment",
            "legal_category": "civil_lease_definition",
            "jeonse_range": (0.42, 0.62),
            "mortgage_range": (0.00, 0.14),
            "senior_range": (0.00, 0.06),
            "gap_range": (-6, 6),
            "flags": {
                "seizure": True,
                "provisional_seizure": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 2,
            "source": "synthetic_counterfactual_danger",
            "text": "부채비율은 낮지만 신탁등기 수탁자 동의 없음 임대 권한 불명 보증보험 불가",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "legal_category": "civil_lease_definition",
            "jeonse_range": (0.48, 0.68),
            "mortgage_range": (0.00, 0.08),
            "senior_range": (0.00, 0.05),
            "gap_range": (-4, 8),
            "flags": {
                "trust_registered": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
                "suspicious_special_clause": True,
            },
        },
        {
            "label": 2,
            "source": "synthetic_counterfactual_danger",
            "text": "전세보증 사고 이력 임대인 다주택 반환 부담 특약 불리 보증보험 불가",
            "contract_type": "jeonse",
            "property_type": "villa",
            "legal_category": "criminal_fraud",
            "jeonse_range": (0.62, 0.82),
            "mortgage_range": (0.02, 0.18),
            "senior_range": (0.00, 0.10),
            "gap_range": (2, 18),
            "flags": {
                "landlord_multiple_properties": True,
                "landlord_prior_incidents": True,
                "suspicious_special_clause": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
            },
        },
        {
            "label": 2,
            "source": "synthetic_counterfactual_danger",
            "text": "실제 지급액과 계약서 보증금 차이 허위계약 보증 대출 사기 의심 시세 괴리 큼",
            "contract_type": "jeonse",
            "property_type": "officetel",
            "legal_category": "criminal_fraud",
            "jeonse_range": (0.68, 0.86),
            "mortgage_range": (0.00, 0.10),
            "senior_range": (0.00, 0.08),
            "gap_range": (24, 46),
            "flags": {
                "broker_advertising_issue": True,
                "suspicious_special_clause": True,
                "guarantee_insurance_available": False,
                "broker_explained_rights": False,
            },
        },
    ]

    for idx in range(count):
        template = templates[idx % len(templates)]
        label = int(template["label"])
        market = round(rng.uniform(130, 1400), 1)
        contract_type = str(template["contract_type"])
        jeonse_ratio = rng.uniform(*template["jeonse_range"])
        mortgage_ratio = rng.uniform(*template["mortgage_range"])
        senior_ratio = rng.uniform(*template["senior_range"])
        if contract_type == "sale":
            sale_price = round(market * rng.uniform(0.88, 1.12), 1)
            deposit = round(sale_price * rng.uniform(0.04, 0.14), 1)
            monthly_rent = 0.0
        elif contract_type == "monthly_rent":
            sale_price = 0.0
            deposit = round(market * jeonse_ratio, 1)
            monthly_rent = round(rng.uniform(0.35, 2.8), 2)
        else:
            sale_price = 0.0
            deposit = round(market * jeonse_ratio, 1)
            monthly_rent = 0.0

        mortgage = round(market * mortgage_ratio, 1)
        senior = round(market * senior_ratio, 1)
        flags = {
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": label >= 1 and rng.random() < 0.24,
            "guarantee_insurance_available": contract_type != "sale" and label == 0,
            "fixed_date_ready": label != 2,
            "move_in_ready": label != 2,
            "broker_explained_rights": label == 0,
        }
        flags.update(template["flags"])
        signals = extract_text_signals(str(template["text"]))
        score = {0: rng.uniform(7, 33), 1: rng.uniform(41, 66), 2: rng.uniform(73, 99)}[label]
        rows.append(
            {
                "source": template["source"],
                "source_case_number": f"COUNTERFACTUAL_{idx + 1:05d}",
                "source_case_name": "추가학습 반례 기반 계약 예시",
                "derivation_variant_id": idx,
                "contract_type": contract_type,
                "property_type": template["property_type"],
                "region": rng.choice(["수도권", "비수도권", "광역시", "지방중소도시"]),
                "deposit_million": deposit,
                "monthly_rent_million": monthly_rent,
                "sale_price_million": sale_price,
                "estimated_market_price_million": market,
                "mortgage_million": mortgage,
                "senior_claim_million": senior,
                "jeonse_ratio": round(deposit / max(market, 1), 4),
                "debt_ratio": round((deposit + mortgage + senior) / max(market, 1), 4),
                "nearby_market_gap_percent": round(rng.uniform(*template["gap_range"]), 2),
                "contract_period_months": rng.choice([0, 12, 24, 36]) if contract_type == "sale" else rng.choice([12, 24, 24, 36]),
                **flags,
                "danger_term_count": signals.danger_term_count,
                "registry_term_count": signals.registry_term_count,
                "broker_term_count": signals.broker_term_count,
                "lease_term_count": signals.lease_term_count,
                "sale_term_count": signals.sale_term_count,
                "has_crime_signal": signals.has_crime_signal,
                "has_broker_signal": signals.has_broker_signal,
                "has_registry_signal": signals.has_registry_signal,
                "legal_category": template["legal_category"],
                "combined_text": str(template["text"]),
                SCORE_COLUMN: round(score, 2),
                TARGET_COLUMN: label,
            }
        )
    return rows


def build_derived_contract_dataset(
    seed: int = 42,
    safe_examples: int = 18000,
    variants_per_case: int = 28,
    hard_examples: int = 10000,
    public_indicator_examples: int = 12000,
    counterfactual_examples: int = 16000,
) -> pd.DataFrame:
    # 최종 학습 데이터 = 판례 파생 + 정상 기준 + 경계/스트레스 + 공개자료 지표 + 추가 반례 예시.
    rng = random.Random(seed)
    case_rows = load_case_rows()
    derived = [
        _contract_features_from_case(row, rng, variant_id=variant_id)
        for row in case_rows
        for variant_id in range(variants_per_case)
    ]
    derived.extend(_safe_synthetic_examples(safe_examples, rng))
    derived.extend(_hard_boundary_examples(hard_examples, rng))
    derived.extend(_public_indicator_examples(public_indicator_examples, rng))
    derived.extend(_counterfactual_stress_examples(counterfactual_examples, rng))
    df = pd.DataFrame(derived)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def write_dataset(df: pd.DataFrame, path: Path = DERIVED_CONTRACTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_public_risk_indicators(path: Path = PUBLIC_RISK_INDICATORS_CSV) -> pd.DataFrame:
    df = pd.DataFrame(PUBLIC_RISK_INDICATORS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def write_external_case_references(path: Path = EXTERNAL_CASE_REFERENCES_CSV) -> pd.DataFrame:
    df = pd.DataFrame(EXTERNAL_CASE_REFERENCES)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def write_data_quality_report(df: pd.DataFrame, path: Path = DATA_DIR / "data_quality_report.json") -> None:
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "source_counts": df["source"].value_counts().to_dict(),
        "label_counts": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "contract_type_counts": df["contract_type"].value_counts().to_dict(),
        "property_type_counts": df["property_type"].value_counts().to_dict(),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "jeonse_ratio_quantiles": df["jeonse_ratio"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).round(4).to_dict(),
        "debt_ratio_quantiles": df["debt_ratio"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).round(4).to_dict(),
        "note": "서비스형 학습 절차를 흉내 내기 위해 판례 기반 다중 변형, 안전 기준 예시, 경계/스트레스 예시, 공식 공개자료 기반 위험 지표 예시, 추가 반례 예시를 결합했다. 실제 피해자 원천 기록은 아니다.",
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def create_manual_scenarios(path: Path = MANUAL_SCENARIOS_CSV) -> pd.DataFrame:
    scenarios = [
        {
            "scenario_id": "REALISTIC_001",
            "name": "신축 빌라 고전세가율 + 근저당 + 보증보험 불가",
            "expected_min_score": 78,
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 270,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 290,
            "mortgage_million": 115,
            "senior_claim_million": 30,
            "seizure": False,
            "provisional_seizure": True,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": True,
            "landlord_prior_incidents": True,
            "broker_unregistered": False,
            "broker_advertising_issue": True,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 18,
            "contract_period_months": 24,
            "special_clause_text": "임차인은 임대인의 담보 제공 및 채권양도에 이의를 제기하지 않는다.",
            "user_situation_text": "사회초년생이 시세보다 높은 보증금의 신축 빌라 전세계약을 앞두고 있다.",
        },
        {
            "scenario_id": "REALISTIC_002",
            "name": "신탁등기 설명 누락 오피스텔",
            "expected_min_score": 74,
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "광역시",
            "deposit_million": 180,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 210,
            "mortgage_million": 20,
            "senior_claim_million": 45,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": True,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 12,
            "contract_period_months": 24,
            "special_clause_text": "신탁원부 확인 없이 임대인이 직접 계약 가능하다고 설명함.",
            "user_situation_text": "등기부에 신탁 표시가 있으나 중개사가 문제없다고만 말한다.",
        },
        {
            "scenario_id": "REALISTIC_003",
            "name": "위반건축물 다가구 + 선순위 보증금 불명",
            "expected_min_score": 70,
            "contract_type": "monthly_rent",
            "property_type": "multi_family",
            "region": "수도권",
            "deposit_million": 90,
            "monthly_rent_million": 0.8,
            "sale_price_million": 0,
            "estimated_market_price_million": 220,
            "mortgage_million": 95,
            "senior_claim_million": 80,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": True,
            "landlord_multiple_properties": True,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 7,
            "contract_period_months": 12,
            "special_clause_text": "선순위 임차보증금 내역은 계약 후 알려주겠다고 함.",
            "user_situation_text": "다가구 주택의 전체 선순위 보증금 확인이 어렵다.",
        },
        {
            "scenario_id": "REALISTIC_004",
            "name": "안전한 아파트 전세",
            "expected_max_score": 38,
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 280,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 520,
            "mortgage_million": 0,
            "senior_claim_million": 0,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": -3,
            "contract_period_months": 24,
            "special_clause_text": "잔금일 전입신고 및 확정일자 진행, 권리변동 시 계약 해제.",
            "user_situation_text": "등기부 권리침해가 없고 보증보험 가입 가능하다.",
        },
        {
            "scenario_id": "REALISTIC_005",
            "name": "매매 잔금 전 가압류 발견",
            "expected_min_score": 66,
            "contract_type": "sale",
            "property_type": "apartment",
            "region": "비수도권",
            "deposit_million": 35,
            "monthly_rent_million": 0,
            "sale_price_million": 410,
            "estimated_market_price_million": 400,
            "mortgage_million": 130,
            "senior_claim_million": 0,
            "seizure": False,
            "provisional_seizure": True,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": False,
            "move_in_ready": False,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": 2,
            "contract_period_months": 0,
            "special_clause_text": "잔금 전 권리침해 말소 약속만 있고 말소 특약이 약함.",
            "user_situation_text": "매매계약 후 잔금 전 등기부에 가압류가 올라온 상황.",
        },
        {
            "scenario_id": "REALISTIC_006",
            "name": "중개보조원 단독 진행 + 허위광고 의심",
            "expected_min_score": 62,
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "지방중소도시",
            "deposit_million": 120,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 150,
            "mortgage_million": 35,
            "senior_claim_million": 10,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": True,
            "landlord_prior_incidents": False,
            "broker_unregistered": True,
            "broker_advertising_issue": True,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 9,
            "contract_period_months": 24,
            "special_clause_text": "광고에는 무융자라고 되어 있었으나 등기부에는 근저당이 있음.",
            "user_situation_text": "공인중개사가 아닌 직원이 계약 설명 대부분을 진행했다.",
        },
        {
            "scenario_id": "REALISTIC_007",
            "name": "실거래가 대비 낮은 전세가율 + 권리침해 없음",
            "expected_max_score": 34,
            "expected_model_grade": "안전",
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 240,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 560,
            "mortgage_million": 0,
            "senior_claim_million": 0,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": -4,
            "contract_period_months": 24,
            "special_clause_text": "잔금 전 등기부 재확인 및 권리침해 발생 시 해제 특약.",
            "user_situation_text": "실거래가와 비교해 보증금이 낮고 보증보험 가입 가능하다.",
        },
        {
            "scenario_id": "REALISTIC_008",
            "name": "보증금 부풀림 의심 전세대출 계약",
            "expected_min_score": 76,
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "광역시",
            "deposit_million": 310,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 260,
            "mortgage_million": 10,
            "senior_claim_million": 20,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": True,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": True,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 32,
            "contract_period_months": 24,
            "special_clause_text": "대출 가능 금액을 맞추기 위해 계약서상 보증금을 높게 적자는 제안.",
            "user_situation_text": "실제 지급 예정 보증금과 계약서 기재 보증금이 다르다.",
        },
        {
            "scenario_id": "REALISTIC_009",
            "name": "신탁등기 + 신탁원부 미확인 + 보증보험 불가",
            "expected_min_score": 78,
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "수도권",
            "deposit_million": 220,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 250,
            "mortgage_million": 0,
            "senior_claim_million": 45,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": True,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 14,
            "contract_period_months": 24,
            "special_clause_text": "신탁원부와 수탁자 동의서는 계약 후 제공한다고 함.",
            "user_situation_text": "등기부에 신탁 표시가 있고 보증보험 가입이 안 된다고 안내받았다.",
        },
        {
            "scenario_id": "REALISTIC_010",
            "name": "다가구 선순위 보증금 확인 완료 저위험 월세",
            "expected_max_score": 42,
            "contract_type": "monthly_rent",
            "property_type": "multi_family",
            "region": "비수도권",
            "deposit_million": 35,
            "monthly_rent_million": 0.55,
            "sale_price_million": 0,
            "estimated_market_price_million": 260,
            "mortgage_million": 15,
            "senior_claim_million": 20,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": 1,
            "contract_period_months": 24,
            "special_clause_text": "선순위 임차보증금 내역을 계약서에 첨부하고 변동 시 해제.",
            "user_situation_text": "다가구지만 전체 선순위 보증금과 근저당 금액을 확인했다.",
        },
        {
            "scenario_id": "REALISTIC_011",
            "name": "낮은 전세가율이지만 갑구 압류가 있는 전세",
            "expected_min_score": 70,
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 210,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 510,
            "mortgage_million": 15,
            "senior_claim_million": 0,
            "seizure": True,
            "provisional_seizure": True,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": -2,
            "contract_period_months": 24,
            "special_clause_text": "압류는 곧 해제된다고만 설명하고 말소 조건은 계약서에 없다.",
            "user_situation_text": "전세가율은 낮지만 등기부 갑구에 압류와 가압류가 함께 표시되어 있다.",
        },
        {
            "scenario_id": "REALISTIC_012",
            "name": "근저당 일부 있지만 말소특약이 명확한 매매",
            "expected_max_score": 45,
            "contract_type": "sale",
            "property_type": "apartment",
            "region": "광역시",
            "deposit_million": 40,
            "monthly_rent_million": 0,
            "sale_price_million": 620,
            "estimated_market_price_million": 630,
            "mortgage_million": 70,
            "senior_claim_million": 0,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": -1,
            "contract_period_months": 0,
            "special_clause_text": "잔금과 동시에 근저당 말소, 말소 불이행 시 계약 해제 및 위약금 지급.",
            "user_situation_text": "시세와 매매가가 비슷하고 갑구 권리침해는 없다.",
        },
        {
            "scenario_id": "REALISTIC_013",
            "name": "전세가율만 높은 보증보험 가능 오피스텔",
            "expected_min_score": 40,
            "expected_max_score": 68,
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "비수도권",
            "deposit_million": 205,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 250,
            "mortgage_million": 0,
            "senior_claim_million": 0,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": False,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": False,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": 8,
            "contract_period_months": 24,
            "special_clause_text": "잔금 전 등기부 재확인, 보증보험 가입 불가 시 계약 해제.",
            "user_situation_text": "권리침해는 없지만 전세가율이 높아 주변 시세를 다시 확인하려 한다.",
        },
        {
            "scenario_id": "REALISTIC_014",
            "name": "보증금은 낮지만 신탁원부 미확인 전세",
            "expected_min_score": 72,
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "수도권",
            "deposit_million": 160,
            "monthly_rent_million": 0,
            "sale_price_million": 0,
            "estimated_market_price_million": 330,
            "mortgage_million": 0,
            "senior_claim_million": 0,
            "seizure": False,
            "provisional_seizure": False,
            "trust_registered": True,
            "illegal_building": False,
            "landlord_multiple_properties": False,
            "landlord_prior_incidents": False,
            "broker_unregistered": False,
            "broker_advertising_issue": False,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 0,
            "contract_period_months": 24,
            "special_clause_text": "수탁자 동의서와 신탁원부는 계약 후 전달한다고 되어 있다.",
            "user_situation_text": "보증금은 낮지만 등기부에 신탁등기가 있고 임대 권한 확인이 되지 않았다.",
        },
    ]
    df = pd.DataFrame(scenarios)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def write_reference_sources(path: Path = REFERENCE_SOURCES_MD) -> None:
    path.write_text(
        """# 오프라인 참고 공개자료와 위험 신호 매핑

이 프로젝트는 실행 시 외부 API에 의존하지 않는다. 아래 자료는 모델 피처와 규칙 설계에 반영한 공개 기준이며, 실제 앱 런타임은 로컬 CSV와 학습 아티팩트만 사용한다.

## 공식 참고 자료

1. 국토교통부 실거래가 공개시스템
   - URL: https://rt.molit.go.kr/pt/info/info.do?mobileAt=v
   - 활용: 주변 실거래가와 입력 보증금/매매가의 괴리율 피처 설계
   - 확인 내용: 부동산 거래신고제를 통해 수집된 실거래 자료를 공개한다.

2. 공공데이터포털 `국토교통부_아파트 매매 실거래가 자료`
   - URL: https://www.data.go.kr/data/15126469/openapi.do
   - 활용: 오프라인 확장 시 지역·기간별 매매 신고정보를 수집할 수 있는 공개 API 후보

3. 공공데이터포털 `국토교통부_건축HUB_건축물대장정보 서비스`
   - URL: https://www.data.go.kr/data/15134735/openapi.do
   - 활용: 위반건축물 여부, 건축물대장 속성 확인 피처의 공개자료 후보

4. 공공데이터포털 `국토교통부_공동주택 기본 정보제공 서비스`
   - URL: https://www.data.go.kr/data/15058453/openapi.do
   - 활용: 공동주택 기본정보와 주변 시설·관리정보 확장 후보

5. HUG 전세보증금반환보증 상품개요
   - URL: https://m.khug.or.kr/hug/web/ig/dr/igdr000001.jsp?tabMenu=Y
   - 활용: 전세보증금+선순위채권이 주택가격×90% 이내인지, 등기부 권리침해사항·선순위채권·신탁/담보 제한 특약을 위험 신호로 반영

6. HUG 전세 및 임대보증 공공데이터
   - URL: https://khug.or.kr/houstar/web/p03/01/p030105.jsp?articleId=34712&currentPage=1&mode=S
   - 활용: 지역별 전세보증사고 현황을 향후 오프라인 데이터로 확장할 수 있는 후보

7. 대법원 인터넷등기소
   - URL: https://www.iros.go.kr
   - 활용: 압류, 가압류, 가처분, 근저당, 신탁 등 등기부 권리침해 신호 설계

8. 국가법령정보센터 판례/법령
   - URL: https://www.law.go.kr
   - 활용: 판례와 법령상 사기, 공인중개사 표시광고, 임대차 보호요건 참조
   - 수집 메모: OPEN API는 등록 IP/도메인 검증이 필요해 로컬 대량 호출은 실패했고, 접근 가능한 공개 페이지와 기존 판례 CSV를 오프라인 참조로 사용했다.

9. 한국경제 대법원 보증채무금 보도
   - URL: https://www.hankyung.com/article/202506220823i
   - 활용: 전세보증금 부풀림/허위 전세계약 위험 유형을 공개자료 기반 파생 사례로 반영

## 모델 피처 반영

- `jeonse_ratio`: 보증금 / 추정 시세
- `debt_ratio`: (보증금 + 근저당 + 선순위채권) / 추정 시세
- `seizure`, `provisional_seizure`: 갑구 권리침해 신호
- `trust_registered`: 신탁등기 및 신탁원부 확인 필요 신호
- `illegal_building`: 건축물대장상 위반건축물 위험
- `broker_unregistered`, `broker_advertising_issue`: 중개사/광고 관련 위험
- `guarantee_insurance_available`: 보증보험 가능 여부
- `public_risk_indicators.csv`: 공식 공개자료 기준 위험 지표
- `external_case_references.csv`: 추가 판례/보도 참조 메타데이터

## 한계

- 실제 등기부등본·건축물대장·실거래가 원문을 자동 조회하지 않는다.
- 국가법령정보 OPEN API는 등록 검증 없이는 현재 로컬에서 직접 대량 호출되지 않았다.
- 파생 계약 예시는 판례와 공개 기준을 바탕으로 만든 학습용 예시이며 실제 피해자 기록이 아니다.
- 출력은 계약 전 위험 신호 탐지이며 법적 판단이 아니다.
""",
        encoding="utf-8",
    )


def build_all_datasets(seed: int = 42) -> dict[str, object]:
    df = build_derived_contract_dataset(seed=seed)
    write_dataset(df)
    write_data_quality_report(df)
    manual = create_manual_scenarios()
    public_indicators = write_public_risk_indicators()
    external_refs = write_external_case_references()
    write_reference_sources()
    return {
        "derived_rows": len(df),
        "manual_scenarios": len(manual),
        "public_risk_indicators": len(public_indicators),
        "external_case_references": len(external_refs),
        "label_counts": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "paths": {
            "derived_contracts": str(DERIVED_CONTRACTS_CSV),
            "manual_scenarios": str(MANUAL_SCENARIOS_CSV),
            "public_risk_indicators": str(PUBLIC_RISK_INDICATORS_CSV),
            "external_case_references": str(EXTERNAL_CASE_REFERENCES_CSV),
            "reference_sources": str(REFERENCE_SOURCES_MD),
        },
    }


if __name__ == "__main__":
    summary = build_all_datasets()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
