from __future__ import annotations

from dataclasses import dataclass


DANGER_TERMS = [
    "사기",
    "기망",
    "편취",
    "횡령",
    "배임",
    "공갈",
    "강박",
    "협박",
    "위조",
    "무고",
    "신탁사기",
    "무자본",
    "깡통전세",
    "보증금 미반환",
    "경매",
    "압류",
    "가압류",
    "가처분",
    "가등기",
    "명의도용",
    "명의 불일치",
    "신분증 불일치",
    "대리계약",
    "위임장 미확인",
    "인감증명 미확인",
    "이중계약",
    "중복계약",
    "당일 근저당",
    "전입 전 근저당",
    "전입신고 지연",
    "대항력 포기",
    "국세 체납",
    "지방세 체납",
    "당해세",
    "선순위보증금 숨김",
    "다중 임차인",
]

REGISTRY_TERMS = [
    "근저당",
    "저당권",
    "담보",
    "전세권",
    "소유권이전",
    "등기",
    "명의신탁",
    "가등기",
    "신탁",
    "신탁원부",
    "수탁자",
    "갑구",
    "을구",
    "말소",
    "말소특약",
    "당일 근저당",
    "전입 전 근저당",
    "국세 체납",
    "지방세 체납",
]

BROKER_TERMS = [
    "공인중개",
    "중개업자",
    "중개보조",
    "중개대상물",
    "확인·설명",
    "표시·광고",
    "공제금",
    "무등록",
    "허위광고",
    "중개보조원",
    "대리계약",
    "위임장",
]

LEASE_TERMS = [
    "전세",
    "월세",
    "임대차",
    "임차보증금",
    "보증금",
    "전세권",
    "전입신고",
    "확정일자",
    "대항력",
    "보증보험",
    "선순위보증금",
]

SALE_TERMS = [
    "매매",
    "매수",
    "매도",
    "매매대금",
    "계약금",
    "중도금",
    "잔금",
    "분양",
]


@dataclass(frozen=True)
class TextSignals:
    danger_term_count: int
    registry_term_count: int
    broker_term_count: int
    lease_term_count: int
    sale_term_count: int
    has_crime_signal: bool
    has_broker_signal: bool
    has_registry_signal: bool


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def extract_text_signals(text: str) -> TextSignals:
    normalized = text or ""
    danger = count_terms(normalized, DANGER_TERMS)
    registry = count_terms(normalized, REGISTRY_TERMS)
    broker = count_terms(normalized, BROKER_TERMS)
    lease = count_terms(normalized, LEASE_TERMS)
    sale = count_terms(normalized, SALE_TERMS)
    return TextSignals(
        danger_term_count=danger,
        registry_term_count=registry,
        broker_term_count=broker,
        lease_term_count=lease,
        sale_term_count=sale,
        has_crime_signal=danger > 0,
        has_broker_signal=broker > 0,
        has_registry_signal=registry > 0,
    )
