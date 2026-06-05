# 테스트 결과

## 자동 테스트

명령:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

결과:

```text
6 passed in 3.09s
```

## 테스트 범위

- 데이터 로딩: 원본 판례 CSV 900건 이상 로딩 확인
- 파생 데이터 생성: 모델 필수 컬럼과 `안전/주의/위험` 라벨 포함 확인
- 모델 구조: `sklearn.ensemble.BaggingClassifier` 사용 확인
- 스코어링: 고위험 입력과 저위험 입력의 점수/등급 차이 확인
- 현실형 사례: 데이터셋에 없는 10개 사용자 시나리오 기준 점수 범위 확인
- 정상 계약 사례: 일부 안전 시나리오에서 Bagging 모델 단독 예측도 `안전`인지 확인

## 현실형 수동 시나리오 결과

상세 결과 파일:

- `학습과정/manual_scenario_predictions.csv`

요약:

| 시나리오 | 결과 |
|---|---|
| 신축 빌라 고전세가율 + 근저당 + 보증보험 불가 | 위험 |
| 신탁등기 설명 누락 오피스텔 | 위험 |
| 위반건축물 다가구 + 선순위 보증금 불명 | 위험 |
| 안전한 아파트 전세 | 안전 |
| 매매 잔금 전 가압류 발견 | 위험 |
| 중개보조원 단독 진행 + 허위광고 의심 | 위험 |
| 실거래가 대비 낮은 전세가율 + 권리침해 없음 | 안전 |
| 보증금 부풀림 의심 전세대출 계약 | 위험 |
| 신탁등기 + 신탁원부 미확인 + 보증보험 불가 | 위험 |
| 다가구 선순위 보증금 확인 완료 저위험 월세 | 안전 |

## API 및 웹 시연 검증

서버:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_web.py --host 127.0.0.1 --port 8765
```

검증 결과:

- `GET /api/health`: 200, `model_exists=true`
- `GET /static/risk-illustration.svg`: 200, `image/svg+xml`
- `POST /api/predict`: 200, 안전 전세 예시 `안전 15.5점`
- 브라우저 콘솔: 메시지 없음
- 네트워크: `/predict` POST 200, `risk-illustration.svg` GET 200

브라우저 캡처:

- `학습과정/web_demo_desktop_final.png`
- `학습과정/web_demo_result_desktop_final.png`
- `학습과정/web_demo_mobile_final.png`
