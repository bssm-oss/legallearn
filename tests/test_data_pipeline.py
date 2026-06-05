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
        user_phrase_examples=36,
        colloquial_examples=36,
        payment_building_examples=36,
        tenancy_title_examples=36,
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
            "synthetic_user_phrase_safe",
            "synthetic_user_phrase_caution",
            "synthetic_user_phrase_danger",
            "synthetic_colloquial_safe",
            "synthetic_colloquial_caution",
            "synthetic_colloquial_danger",
            "synthetic_payment_building_safe",
            "synthetic_payment_building_caution",
            "synthetic_payment_building_danger",
            "synthetic_tenancy_title_safe",
            "synthetic_tenancy_title_caution",
            "synthetic_tenancy_title_danger",
        ]
    ).all()
    assert df["source"].str.startswith("synthetic_counterfactual").any()
    assert df["source"].str.startswith("synthetic_emerging").any()
    assert df["source"].str.startswith("synthetic_user_phrase").any()
    assert df["source"].str.startswith("synthetic_colloquial").any()
    assert df["source"].str.startswith("synthetic_payment_building").any()
    assert df["source"].str.startswith("synthetic_tenancy_title").any()
    assert df["safety_term_count"].max() > 0
    assert df["critical_term_count"].max() > 0
