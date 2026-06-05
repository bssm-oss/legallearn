from risk_detector.data.pipeline import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_derived_contract_dataset,
    load_case_rows,
)


def test_load_case_rows():
    rows = load_case_rows()
    assert len(rows) >= 900
    assert {"case_number", "summary", "legal_category"}.issubset(rows[0].keys())


def test_build_derived_contract_dataset_has_required_columns():
    df = build_derived_contract_dataset(
        seed=7,
        safe_examples=30,
        variants_per_case=2,
        hard_examples=30,
        public_indicator_examples=30,
        counterfactual_examples=30,
        emerging_examples=36,
    )
    assert len(df) >= 900
    for column in MODEL_FEATURES + [TARGET_COLUMN]:
        assert column in df.columns
    assert set(df[TARGET_COLUMN].unique()).issuperset({0, 1, 2})
    assert df["source"].isin(
        [
            "case_derived",
            "synthetic_safe_reference",
            "synthetic_hard_danger",
            "synthetic_hard_caution",
            "synthetic_hard_safe",
            "public_indicator_safe",
            "public_indicator_caution",
            "public_indicator_danger",
            "synthetic_counterfactual_safe",
            "synthetic_counterfactual_caution",
            "synthetic_counterfactual_danger",
            "synthetic_emerging_safe",
            "synthetic_emerging_caution",
            "synthetic_emerging_danger",
        ]
    ).all()
    assert df["source"].str.startswith("synthetic_counterfactual").any()
    assert df["source"].str.startswith("synthetic_emerging").any()
