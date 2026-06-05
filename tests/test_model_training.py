from sklearn.ensemble import BaggingClassifier

from risk_detector.model.training import build_bagging_pipeline


def test_pipeline_uses_sklearn_bagging_classifier():
    model = build_bagging_pipeline(n_estimators=3)
    assert isinstance(model.named_steps["bagging"], BaggingClassifier)
    assert model.named_steps["bagging"].n_estimators == 3

