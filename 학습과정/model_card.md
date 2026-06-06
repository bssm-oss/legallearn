# 모델 카드

## 모델 개요

- 이름: 자동화된 부동산 기망·사기 및 불법 행위 탐지 Bagging 모델
- 알고리즘: `sklearn.ensemble.BaggingClassifier`
- 기본 추정기: `sklearn.tree.DecisionTreeClassifier`
- 학습 데이터: `/Users/heodongun/Desktop/크롤링/학습/데이터/derived_contract_cases.csv`
- 전체 행 수: 132488
- 학습/검증/홀드아웃: 86166 / 19289 / 27033

## 입력과 출력

- 입력: 계약 유형, 보증금/시세/근저당/선순위채권, 등기부·건축물·중개 위험 신호, 특약·상황 텍스트
- 출력: `안전`, `주의`, `위험` 3개 등급의 모델 확률
- 최종 앱 점수: Bagging 모델 확률과 명시적 위험 규칙을 혼합한 0~100점

## 추가학습 구성

- 학습 프로필: `document_mismatch_text_robust_offline_demo`
- 보강 전략: Added counterfactual, emerging field-pattern, natural-language user phrase, colloquial text-only, payment/building text-only, tenancy/title text-only, priority/auction text-only, and document mismatch text-only examples; introduced explicit safety and critical-risk text signals for verified proxy contracts, protection requirements, clean registry cases, trust, seizure, delayed move-in, tax arrears, double-contract, no-guarantee, account mismatch, illegal building, pressure-to-pay, unauthorized sublease, lease-registration, unregistered or pre-approval new building, non-refundable reservation deposit, tenant-registry disclosure refusal, senior tenant deposit unknown, auction or public-auction notices, ownership transfer during contract, same-day loan-before-move-in, land-right unregistered, registry/building/contract unit mismatch, free-residence confirmation, jeonse-right refusal, corporate authority documents, and mortgage cancellation-before-payment patterns; kept grouped validation.

## 평가 결과

Validation:

- Accuracy: 0.9927
- Balanced Accuracy: 0.9923
- Macro F1: 0.9926
- Rows: 19289

Holdout:

- Accuracy: 0.9807
- Balanced Accuracy: 0.9791
- Macro F1: 0.9797
- Rows: 27033

## 사용 범위

이 모델은 계약 전 2차 확인용 위험 신호 탐지 모델이다. 사기 여부, 책임 소재, 계약 취소 가능성에 대한 법적 판단을 대신하지 않는다.

## 주요 한계

1. 실제 피해자 원천 계약 데이터가 아니라 판례 기반 파생 데이터와 공개 기준 기반 합성 예시로 학습했다.
2. 런타임에서 실거래가, 등기부등본, 건축물대장, 중개사 등록 정보를 실시간 조회하지 않는다.
3. 실제 서비스 적용 전에는 실제 신고/보증사고/정상계약 데이터로 재학습과 외부 검증이 필요하다.
