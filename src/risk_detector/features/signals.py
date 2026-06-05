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
    "계좌 명의",
    "다른 사람 계좌",
    "대표 개인계좌",
    "개인계좌",
    "현금 요구",
    "계약금 먼저",
    "등기부는 나중",
    "불법증축",
    "쪼개기",
    "무허가",
    "용도위반",
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

SAFETY_TERMS = [
    "확인 완료",
    "원본 확인",
    "권리침해 없음",
    "보증보험 가능",
    "확정일자 진행",
    "전입신고 가능",
    "전입 가능",
    "말소특약",
    "계약 해제",
    "소유자 영상통화",
    "위임장 원본",
    "인감증명 원본",
    "선순위보증금 명세",
    "전체 선순위",
    "낮은 전세가율",
    "낮은 부채비율",
    "정상 계약",
    "보호요건 충족",
    "등기부 깨끗",
    "갑구 을구 깨끗",
    "본인 확인",
    "신분증 일치",
    "전입 바로 가능",
    "서류 원본 확인",
    "확정일자 바로",
    "임대인 계좌 일치",
    "계약금 보류",
    "건축물대장 확인 완료",
    "건축물대장 정상",
    "위반건축물 아님",
]

CRITICAL_RISK_TERMS = [
    "보증보험 불가",
    "신탁원부",
    "수탁자 동의",
    "임대 권한 불명",
    "압류",
    "가압류",
    "가처분",
    "위반건축물",
    "선순위보증금 불명",
    "선순위보증금 숨김",
    "명의 불일치",
    "신분증 불일치",
    "전입신고 지연",
    "전입 전 근저당",
    "당일 근저당",
    "대항력 공백",
    "국세 체납",
    "지방세 체납",
    "당해세",
    "이중계약",
    "중복계약",
    "보증금 부풀림",
    "허위계약",
    "압류 곧",
    "곧 풀린다",
    "신탁원부 없어도",
    "수탁자 동의 나중",
    "보증보험 안",
    "보증보험 안됨",
    "계약 먼저",
    "등기부랑 광고 다름",
    "집주인 이름 다름",
    "계좌 명의",
    "다른 사람 계좌",
    "임대인 이름이 아닙",
    "대표 개인계좌",
    "개인계좌로 보내",
    "현금 요구",
    "계약금 먼저",
    "오늘 계약금",
    "빨리 입금",
    "등기부는 나중",
    "건축물대장 못",
    "불법증축",
    "쪼개기",
    "무허가",
    "용도위반",
]


@dataclass(frozen=True)
class TextSignals:
    danger_term_count: int
    registry_term_count: int
    broker_term_count: int
    lease_term_count: int
    sale_term_count: int
    safety_term_count: int
    critical_term_count: int
    has_crime_signal: bool
    has_broker_signal: bool
    has_registry_signal: bool
    has_safety_signal: bool
    has_critical_signal: bool


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def extract_text_signals(text: str) -> TextSignals:
    normalized = text or ""
    danger = count_terms(normalized, DANGER_TERMS)
    registry = count_terms(normalized, REGISTRY_TERMS)
    broker = count_terms(normalized, BROKER_TERMS)
    lease = count_terms(normalized, LEASE_TERMS)
    sale = count_terms(normalized, SALE_TERMS)
    safety = count_terms(normalized, SAFETY_TERMS)
    critical = count_terms(normalized, CRITICAL_RISK_TERMS)
    return TextSignals(
        danger_term_count=danger,
        registry_term_count=registry,
        broker_term_count=broker,
        lease_term_count=lease,
        sale_term_count=sale,
        safety_term_count=safety,
        critical_term_count=critical,
        has_crime_signal=danger > 0,
        has_broker_signal=broker > 0,
        has_registry_signal=registry > 0,
        has_safety_signal=safety > 0,
        has_critical_signal=critical > 0,
    )
