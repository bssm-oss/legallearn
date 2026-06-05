import pandas as pd

from risk_detector.paths import MANUAL_SCENARIOS_CSV
from risk_detector.risk.scorer import RiskScorer


def test_realistic_out_of_dataset_scenarios_match_expected_bounds():
    scorer = RiskScorer()
    scenarios = pd.read_csv(MANUAL_SCENARIOS_CSV, encoding="utf-8-sig")
    assert len(scenarios) >= 6
    for _, row in scenarios.iterrows():
        payload = row.dropna().to_dict()
        result = scorer.score(payload)
        if "expected_min_score" in payload:
            assert result["risk_score"] >= float(payload["expected_min_score"]) - 8, payload["name"]
        if "expected_max_score" in payload:
            assert result["risk_score"] <= float(payload["expected_max_score"]) + 8, payload["name"]
        if payload.get("expected_model_grade"):
            assert result["model_predicted_grade"] == payload["expected_model_grade"], payload["name"]
