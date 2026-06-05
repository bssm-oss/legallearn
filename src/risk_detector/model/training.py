from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from risk_detector.data.pipeline import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SCORE_COLUMN,
    TARGET_COLUMN,
    TEXT_FEATURE,
    build_all_datasets,
)
from risk_detector.paths import DERIVED_CONTRACTS_CSV, LEARNING_DIR, MANUAL_SCENARIOS_CSV, METADATA_PATH, MODEL_PATH
from risk_detector.schemas import RISK_LABELS


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # scikit-learn 파이프라인에 들어가기 전 타입을 고정해 재학습과 예측 입력 형식을 맞춘다.
    prepared = df.copy()
    for col in NUMERIC_FEATURES:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)
    for col in BOOLEAN_FEATURES:
        prepared[col] = prepared[col].astype(bool).astype(int)
    for col in CATEGORICAL_FEATURES:
        prepared[col] = prepared[col].fillna("unknown").astype(str)
    prepared[TEXT_FEATURE] = prepared[TEXT_FEATURE].fillna("").astype(str)
    return prepared


def build_bagging_pipeline(random_state: int = 42, n_estimators: int = 600) -> Pipeline:
    # 수치, 범주, 불리언, 텍스트를 각각 전처리한 뒤 BaggingClassifier가 함께 학습한다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    text_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=4200,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            ("text", text_pipeline, TEXT_FEATURE),
        ]
    )
    tree = DecisionTreeClassifier(
        max_depth=16,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=random_state,
    )
    classifier = BaggingClassifier(
        estimator=tree,
        n_estimators=n_estimators,
        max_samples=0.88,
        max_features=0.94,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("bagging", classifier),
        ]
    )


def load_training_frame(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not DERIVED_CONTRACTS_CSV.exists() or not MANUAL_SCENARIOS_CSV.exists():
        build_all_datasets()
    return pd.read_csv(DERIVED_CONTRACTS_CSV, encoding="utf-8-sig")


def metric_summary(model: Pipeline, x_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, object]:
    y_pred = model.predict(x_eval)
    metrics = {
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_eval, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_eval, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_eval, y_pred, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_eval, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_eval, y_pred, average="macro", zero_division=0)), 4),
        "rows": int(len(x_eval)),
    }
    metrics["confusion_matrix"] = confusion_matrix(y_eval, y_pred, labels=[0, 1, 2]).tolist()
    return metrics


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    learning_dir: Path = LEARNING_DIR,
    random_state: int = 42,
) -> dict[str, object]:
    # holdout 결과뿐 아니라 제출용 산출물인 리포트, 혼동행렬, 중요도, 예측 예시를 파일로 남긴다.
    learning_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(x_test)
    labels = [0, 1, 2]
    target_names = [RISK_LABELS[idx] for idx in labels]

    metrics = metric_summary(model, x_test, y_test)
    metrics["test_rows"] = metrics["rows"]

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    (learning_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{name}" for name in target_names], columns=[f"pred_{name}" for name in target_names])
    cm_df.to_csv(learning_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    try:
        importance_x = x_test.sample(min(len(x_test), 1400), random_state=random_state)
        importance_y = y_test.loc[importance_x.index]
        importance = permutation_importance(
            model,
            importance_x,
            importance_y,
            n_repeats=4,
            random_state=random_state,
            n_jobs=-1,
            scoring="f1_macro",
        )
        importance_df = pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    except Exception as exc:  # pragma: no cover - diagnostic fallback only
        importance_df = pd.DataFrame(
            [
                {
                    "feature": "permutation_importance_error",
                    "importance_mean": 0.0,
                    "importance_std": 0.0,
                    "note": str(exc),
                }
            ]
        )
    importance_df.to_csv(learning_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    examples = x_test.copy()
    examples["actual_label"] = y_test.values
    examples["predicted_label"] = y_pred
    proba = model.predict_proba(x_test)
    for class_idx, class_name in RISK_LABELS.items():
        examples[f"prob_{class_name}"] = proba[:, list(model.classes_).index(class_idx)]
    examples[
        [
            "contract_type",
            "property_type",
            "deposit_million",
            "estimated_market_price_million",
            "debt_ratio",
            "actual_label",
            "predicted_label",
            "prob_안전",
            "prob_주의",
            "prob_위험",
        ]
    ].head(40).to_csv(learning_dir / "prediction_examples.csv", index=False, encoding="utf-8-sig")

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def write_training_report(
    metrics: dict[str, object],
    validation_metrics: dict[str, object],
    df: pd.DataFrame,
    metadata: dict[str, object],
    learning_dir: Path = LEARNING_DIR,
    model_path: Path = MODEL_PATH,
) -> None:
    label_counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()
    label_counts_ko = {RISK_LABELS[int(k)]: int(v) for k, v in label_counts.items()}
    source_counts = df["source"].value_counts().to_dict()
    contract_counts = df["contract_type"].value_counts().to_dict()
    property_counts = df["property_type"].value_counts().to_dict()
    risk_quantiles = df[SCORE_COLUMN].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).round(2).to_dict()
    report = f"""# 학습 과정 보고서

## 목적

이 프로젝트는 전세·월세·매매 계약 전 단계에서 사용자가 입력한 계약 정보를 바탕으로 위험 신호를 탐지하는 오프라인 AI 시연 시스템이다. 출력은 법적 판단이 아니라 계약 전 2차 확인을 위한 위험도와 근거다.

## 사용 데이터

- 원본 판례 데이터: `데이터/real_estate_fraud_cases_filtered.csv`
- 파생 계약 학습 데이터: `데이터/derived_contract_cases.csv`
- 현실형 수동 테스트 시나리오: `데이터/manual_test_scenarios.csv`
- 오프라인 공개자료 기준 문서: `데이터/offline_reference_sources.md`

원본 판례는 사건명, 판시사항, 판결요지, 조문, 키워드, 법률 카테고리를 포함한다. 직접적인 계약별 표준 라벨 데이터가 아니므로, 판례의 위험 유형과 HUG/국토부 공개 기준에서 추출한 위험 신호를 구조화해 학습용 계약 예시로 변환했다.

## 서비스형 학습 구성

- 전체 학습 데이터 행 수: {metadata['rows']}
- 학습 행 수: {metadata['train_rows']}
- 검증 행 수: {metadata['validation_rows']}
- 홀드아웃 테스트 행 수: {metadata['holdout_rows']}
- 파생 방식: 판례 1건당 여러 계약 조건 변형 + 정상 계약 기준 예시 + 위험 경계/스트레스 예시 + 추가 반례 예시 + 신원/대항력/체납/이중계약 현장패턴 예시 + 사용자 자연어 입력 변형 예시 + 구어체 텍스트-only 입력 예시 + 계좌·계약금·건축물 텍스트-only 예시 + 전대차·임차권등기·미등기·가계약금 텍스트-only 예시 + 전입세대·선순위·경매공매·소유자변경 텍스트-only 예시
- 누수 방지: 같은 `source_case_number` 그룹이 학습/검증/홀드아웃에 동시에 들어가지 않도록 분리

데이터 소스 분포:

{json.dumps(source_counts, ensure_ascii=False, indent=2)}

계약 유형 분포:

{json.dumps(contract_counts, ensure_ascii=False, indent=2)}

주택 유형 분포:

{json.dumps(property_counts, ensure_ascii=False, indent=2)}

휴리스틱 위험 점수 분위수:

{json.dumps({str(k): float(v) for k, v in risk_quantiles.items()}, ensure_ascii=False, indent=2)}

## Bagging 알고리즘 적용

- 라이브러리: `scikit-learn`
- 모델: `sklearn.ensemble.BaggingClassifier`
- 기본 추정기: `DecisionTreeClassifier`
- 입력 피처: 전세가율, 부채비율, 근저당, 압류/가압류, 신탁, 위반건축물, 중개사 위험, 보증보험 가능 여부, 판례 텍스트 신호 등
- 출력: `0=안전`, `1=주의`, `2=위험`

Bagging은 여러 개의 결정트리를 bootstrap 표본으로 학습하고 예측을 집계하므로, 단일 트리보다 과적합을 줄이고 여러 위험 신호 조합을 안정적으로 반영하는 데 적합하다.

## 데이터 분포

{json.dumps(label_counts_ko, ensure_ascii=False, indent=2)}

## 평가 결과

- Accuracy: {metrics['accuracy']}
- Balanced Accuracy: {metrics['balanced_accuracy']}
- Macro F1: {metrics['macro_f1']}
- Weighted F1: {metrics['weighted_f1']}
- Macro Precision: {metrics['macro_precision']}
- Macro Recall: {metrics['macro_recall']}
- Test rows: {metrics['test_rows']}

## 검증 분리 방식

- 같은 판례 번호에서 파생된 변형 데이터가 학습/검증/테스트에 동시에 들어가지 않도록 `source_case_number` 기준 그룹 분리를 적용했다.
- Validation rows: {validation_metrics['rows']}
- Validation Macro F1: {validation_metrics['macro_f1']}
- Holdout rows: {metrics['test_rows']}

상세 결과:
- `학습과정/classification_report.txt`
- `학습과정/confusion_matrix.csv`
- `학습과정/feature_importance.csv`
- `학습과정/prediction_examples.csv`
- `학습과정/model_card.md`
- `학습과정/training_audit.json`

## 저장된 모델

- `{model_path}`
- `models/metadata.json`

## 한계

1. 원본 데이터는 판례 중심 데이터이며 실제 계약서 원천 데이터가 아니다.
2. 파생 계약 예시는 학습과 시연을 위한 구조화 예시이며 실제 피해자 기록이 아니다.
3. 등기부등본·건축물대장·실거래가를 런타임에 실시간 조회하지 않는다.
4. 결과는 위험 신호 탐지이며 수사기관·법원의 법적 판단을 대체하지 않는다.
"""
    (learning_dir / "training_report.md").write_text(report, encoding="utf-8")


def write_model_card(metadata: dict[str, object], learning_dir: Path = LEARNING_DIR) -> None:
    metrics = metadata["metrics"]
    validation = metadata["validation_metrics"]
    card = f"""# 모델 카드

## 모델 개요

- 이름: 자동화된 부동산 기망·사기 및 불법 행위 탐지 Bagging 모델
- 알고리즘: `{metadata['algorithm']}`
- 기본 추정기: `{metadata['base_estimator']}`
- 학습 데이터: `{metadata['source_dataset']}`
- 전체 행 수: {metadata['rows']}
- 학습/검증/홀드아웃: {metadata['train_rows']} / {metadata['validation_rows']} / {metadata['holdout_rows']}

## 입력과 출력

- 입력: 계약 유형, 보증금/시세/근저당/선순위채권, 등기부·건축물·중개 위험 신호, 특약·상황 텍스트
- 출력: `안전`, `주의`, `위험` 3개 등급의 모델 확률
- 최종 앱 점수: Bagging 모델 확률과 명시적 위험 규칙을 혼합한 0~100점

## 추가학습 구성

- 학습 프로필: `{metadata.get('training_profile', 'unknown')}`
- 보강 전략: {metadata.get('additional_training_strategy', '판례 파생 데이터와 공개 기준 기반 예시를 결합했다.')}

## 평가 결과

Validation:

- Accuracy: {validation['accuracy']}
- Balanced Accuracy: {validation['balanced_accuracy']}
- Macro F1: {validation['macro_f1']}
- Rows: {validation['rows']}

Holdout:

- Accuracy: {metrics['accuracy']}
- Balanced Accuracy: {metrics['balanced_accuracy']}
- Macro F1: {metrics['macro_f1']}
- Rows: {metrics['test_rows']}

## 사용 범위

이 모델은 계약 전 2차 확인용 위험 신호 탐지 모델이다. 사기 여부, 책임 소재, 계약 취소 가능성에 대한 법적 판단을 대신하지 않는다.

## 주요 한계

1. 실제 피해자 원천 계약 데이터가 아니라 판례 기반 파생 데이터와 공개 기준 기반 합성 예시로 학습했다.
2. 런타임에서 실거래가, 등기부등본, 건축물대장, 중개사 등록 정보를 실시간 조회하지 않는다.
3. 실제 서비스 적용 전에는 실제 신고/보증사고/정상계약 데이터로 재학습과 외부 검증이 필요하다.
"""
    (learning_dir / "model_card.md").write_text(card, encoding="utf-8")


def write_manual_scenario_predictions(learning_dir: Path = LEARNING_DIR) -> None:
    # 재학습할 때마다 사용자형 수동 시나리오 예측 결과를 함께 갱신해 문서와 테스트 산출물이 어긋나지 않게 한다.
    from risk_detector.risk.scorer import RiskScorer

    if not MANUAL_SCENARIOS_CSV.exists() or not MODEL_PATH.exists():
        return
    scenarios = pd.read_csv(MANUAL_SCENARIOS_CSV, encoding="utf-8-sig")
    scorer = RiskScorer()
    rows: list[dict[str, object]] = []
    for _, row in scenarios.iterrows():
        payload = row.dropna().to_dict()
        result = scorer.score(payload)
        probabilities = result["model_probabilities"]
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "name": row["name"],
                "risk_score": result["risk_score"],
                "risk_grade": result["risk_grade"],
                "model_predicted_grade": result["model_predicted_grade"],
                "prob_safe": probabilities.get("안전", 0.0),
                "prob_caution": probabilities.get("주의", 0.0),
                "prob_danger": probabilities.get("위험", 0.0),
            }
        )
    pd.DataFrame(rows).to_csv(learning_dir / "manual_scenario_predictions.csv", index=False, encoding="utf-8-sig")


def train_model(
    rebuild_data: bool = True,
    random_state: int = 42,
    n_estimators: int = 600,
) -> dict[str, object]:
    df = load_training_frame(rebuild=rebuild_data)
    df = _prepare_features(df)
    x = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)
    fallback_groups = pd.Series(df.index.astype(str), index=df.index)
    groups = df["source_case_number"].where(df["source_case_number"].notna(), fallback_groups).astype(str)
    # 같은 판례/공개자료 기준에서 파생된 행이 train과 holdout에 동시에 들어가면 성능이 과대평가된다.
    # 그래서 source_case_number 기준 그룹 분리로 더 엄격한 검증 구조를 사용한다.
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_val_idx, test_idx = next(group_split.split(x, y, groups=groups))
    x_train_val = x.iloc[train_val_idx].reset_index(drop=True)
    y_train_val = y.iloc[train_val_idx].reset_index(drop=True)
    groups_train_val = groups.iloc[train_val_idx].reset_index(drop=True)
    validation_split = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=random_state + 1)
    train_idx, val_idx = next(validation_split.split(x_train_val, y_train_val, groups=groups_train_val))
    x_train = x_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    x_val = x_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]
    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]
    model = build_bagging_pipeline(random_state=random_state, n_estimators=n_estimators)
    model.fit(x_train, y_train)
    validation_metrics = metric_summary(model, x_val, y_val)
    metrics = evaluate_model(model, x_test, y_test, random_state=random_state)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sklearn.ensemble.BaggingClassifier",
        "base_estimator": "sklearn.tree.DecisionTreeClassifier",
        "training_profile": "priority_auction_text_robust_offline_demo",
        "additional_training_strategy": "Added counterfactual, emerging field-pattern, natural-language user phrase, colloquial text-only, payment/building text-only, tenancy/title text-only, and priority/auction text-only examples; introduced explicit safety and critical-risk text signals for verified proxy contracts, protection requirements, clean registry cases, trust, seizure, delayed move-in, tax arrears, double-contract, no-guarantee, account mismatch, illegal building, pressure-to-pay, unauthorized sublease, lease-registration, unregistered or pre-approval new building, non-refundable reservation deposit, tenant-registry disclosure refusal, senior tenant deposit unknown, auction or public-auction notices, ownership transfer during contract, and same-day loan-before-move-in patterns; increased Bagging ensemble capacity while keeping grouped validation.",
        "random_state": random_state,
        "n_estimators": n_estimators,
        "model_path": str(MODEL_PATH),
        "source_dataset": str(DERIVED_CONTRACTS_CSV),
        "manual_scenarios": str(MANUAL_SCENARIOS_CSV),
        "rows": int(len(df)),
        "split_strategy": "GroupShuffleSplit by source_case_number; train/validation/holdout groups are separated",
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "holdout_rows": int(len(x_test)),
        "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()},
        "label_counts": {RISK_LABELS[int(k)]: int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().to_dict().items()},
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
        "text_feature": TEXT_FEATURE,
        "label_mapping": RISK_LABELS,
        "validation_metrics": validation_metrics,
        "metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (LEARNING_DIR / "metrics.json").write_text(
        json.dumps({"validation": validation_metrics, "holdout": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (LEARNING_DIR / "training_audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_training_report(metrics, validation_metrics, df, metadata)
    write_model_card(metadata)
    write_manual_scenario_predictions()
    return metadata


if __name__ == "__main__":
    result = train_model()
    print(json.dumps({"model": result["model_path"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
