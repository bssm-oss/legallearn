from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "데이터"
LEARNING_DIR = PROJECT_ROOT / "학습과정"
MODELS_DIR = PROJECT_ROOT / "models"

SOURCE_CASES_CSV = DATA_DIR / "real_estate_fraud_cases_filtered.csv"
DERIVED_CONTRACTS_CSV = DATA_DIR / "derived_contract_cases.csv"
MANUAL_SCENARIOS_CSV = DATA_DIR / "manual_test_scenarios.csv"
REFERENCE_SOURCES_MD = DATA_DIR / "offline_reference_sources.md"
PUBLIC_RISK_INDICATORS_CSV = DATA_DIR / "public_risk_indicators.csv"
EXTERNAL_CASE_REFERENCES_CSV = DATA_DIR / "external_case_references.csv"

MODEL_PATH = MODELS_DIR / "bagging_risk_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
