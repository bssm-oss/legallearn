# 학습 과정 보고서

## 목적

이 프로젝트는 전세·월세·매매 계약 전 단계에서 사용자가 입력한 계약 정보를 바탕으로 위험 신호를 탐지하는 오프라인 AI 시연 시스템이다. 출력은 법적 판단이 아니라 계약 전 2차 확인을 위한 위험도와 근거다.

## 사용 데이터

- 원본 판례 데이터: `데이터/real_estate_fraud_cases_filtered.csv`
- 파생 계약 학습 데이터: `데이터/derived_contract_cases.csv`
- 현실형 수동 테스트 시나리오: `데이터/manual_test_scenarios.csv`
- 오프라인 공개자료 기준 문서: `데이터/offline_reference_sources.md`

원본 판례는 사건명, 판시사항, 판결요지, 조문, 키워드, 법률 카테고리를 포함한다. 직접적인 계약별 표준 라벨 데이터가 아니므로, 판례의 위험 유형과 HUG/국토부 공개 기준에서 추출한 위험 신호를 구조화해 학습용 계약 예시로 변환했다.

## 서비스형 학습 구성

- 전체 학습 데이터 행 수: 123488
- 학습 행 수: 81099
- 검증 행 수: 17885
- 홀드아웃 테스트 행 수: 24504
- 파생 방식: 판례 1건당 여러 계약 조건 변형 + 정상 계약 기준 예시 + 위험 경계/스트레스 예시 + 추가 반례 예시 + 신원/대항력/체납/이중계약 현장패턴 예시 + 사용자 자연어 입력 변형 예시 + 구어체 텍스트-only 입력 예시 + 계좌·계약금·건축물 텍스트-only 예시
- 누수 방지: 같은 `source_case_number` 그룹이 학습/검증/홀드아웃에 동시에 들어가지 않도록 분리

데이터 소스 분포:

{
  "case_derived": 26488,
  "synthetic_safe_reference": 18000,
  "synthetic_emerging_danger": 9000,
  "public_indicator_danger": 7200,
  "synthetic_counterfactual_danger": 6400,
  "synthetic_counterfactual_safe": 4800,
  "synthetic_counterfactual_caution": 4800,
  "synthetic_user_phrase_danger": 4500,
  "synthetic_emerging_caution": 4500,
  "synthetic_emerging_safe": 4500,
  "synthetic_colloquial_danger": 4305,
  "synthetic_hard_danger": 3334,
  "synthetic_hard_caution": 3334,
  "synthetic_payment_building_danger": 3332,
  "synthetic_hard_safe": 3332,
  "public_indicator_caution": 2400,
  "public_indicator_safe": 2400,
  "synthetic_user_phrase_caution": 2250,
  "synthetic_user_phrase_safe": 2250,
  "synthetic_colloquial_safe": 1848,
  "synthetic_colloquial_caution": 1847,
  "synthetic_payment_building_caution": 1334,
  "synthetic_payment_building_safe": 1334
}

계약 유형 분포:

{
  "jeonse": 69448,
  "sale": 28437,
  "monthly_rent": 25603
}

주택 유형 분포:

{
  "apartment": 39097,
  "villa": 34583,
  "officetel": 23239,
  "multi_family": 23075,
  "commercial": 3494
}

휴리스틱 위험 점수 분위수:

{
  "0.0": 6.0,
  "0.25": 28.02,
  "0.5": 56.18,
  "0.75": 83.39,
  "0.9": 94.53,
  "0.99": 98.14,
  "1.0": 99.0
}

## Bagging 알고리즘 적용

- 라이브러리: `scikit-learn`
- 모델: `sklearn.ensemble.BaggingClassifier`
- 기본 추정기: `DecisionTreeClassifier`
- 입력 피처: 전세가율, 부채비율, 근저당, 압류/가압류, 신탁, 위반건축물, 중개사 위험, 보증보험 가능 여부, 판례 텍스트 신호 등
- 출력: `0=안전`, `1=주의`, `2=위험`

Bagging은 여러 개의 결정트리를 bootstrap 표본으로 학습하고 예측을 집계하므로, 단일 트리보다 과적합을 줄이고 여러 위험 신호 조합을 안정적으로 반영하는 데 적합하다.

## 데이터 분포

{
  "안전": 38940,
  "주의": 36649,
  "위험": 47899
}

## 평가 결과

- Accuracy: 0.9837
- Balanced Accuracy: 0.983
- Macro F1: 0.9826
- Weighted F1: 0.9837
- Macro Precision: 0.9824
- Macro Recall: 0.983
- Test rows: 24504

## 검증 분리 방식

- 같은 판례 번호에서 파생된 변형 데이터가 학습/검증/테스트에 동시에 들어가지 않도록 `source_case_number` 기준 그룹 분리를 적용했다.
- Validation rows: 17885
- Validation Macro F1: 0.995
- Holdout rows: 24504

상세 결과:
- `학습과정/classification_report.txt`
- `학습과정/confusion_matrix.csv`
- `학습과정/feature_importance.csv`
- `학습과정/prediction_examples.csv`
- `학습과정/model_card.md`
- `학습과정/training_audit.json`

## 저장된 모델

- `/Users/heodongun/Desktop/크롤링/학습/models/bagging_risk_model.joblib`
- `models/metadata.json`

## 한계

1. 원본 데이터는 판례 중심 데이터이며 실제 계약서 원천 데이터가 아니다.
2. 파생 계약 예시는 학습과 시연을 위한 구조화 예시이며 실제 피해자 기록이 아니다.
3. 등기부등본·건축물대장·실거래가를 런타임에 실시간 조회하지 않는다.
4. 결과는 위험 신호 탐지이며 수사기관·법원의 법적 판단을 대체하지 않는다.
