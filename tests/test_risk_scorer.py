from risk_detector.risk.scorer import RiskScorer, contract_to_model_row, payload_to_contract


def test_contract_to_model_row_computes_ratios():
    contract = payload_to_contract(
        {
            "contract_type": "jeonse",
            "deposit_million": 270,
            "estimated_market_price_million": 300,
            "mortgage_million": 60,
            "senior_claim_million": 20,
            "guarantee_insurance_available": False,
        }
    )
    row = contract_to_model_row(contract)
    assert row["jeonse_ratio"] == 0.9
    assert row["debt_ratio"] > 1.1


def test_trained_scorer_flags_high_and_low_risk():
    scorer = RiskScorer()
    high = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 270,
            "estimated_market_price_million": 290,
            "mortgage_million": 115,
            "senior_claim_million": 30,
            "provisional_seizure": True,
            "landlord_prior_incidents": True,
            "broker_advertising_issue": True,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "broker_explained_rights": False,
            "nearby_market_gap_percent": 18,
            "special_clause_text": "채권양도와 담보 제공에 이의를 제기하지 않는다.",
        }
    )
    low = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 240,
            "estimated_market_price_million": 520,
            "mortgage_million": 0,
            "senior_claim_million": 0,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": -2,
        }
    )
    assert high["risk_score"] >= 70
    assert high["risk_grade"] == "위험"
    assert low["risk_score"] < 45
    assert low["risk_grade"] in {"안전", "주의"}


def test_counterfactual_boundary_inputs_are_not_under_scored():
    scorer = RiskScorer()
    seizure_with_low_ratio = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 210,
            "estimated_market_price_million": 510,
            "mortgage_million": 15,
            "seizure": True,
            "provisional_seizure": True,
            "guarantee_insurance_available": False,
            "broker_explained_rights": False,
            "special_clause_text": "압류는 곧 해제된다고만 설명하고 말소 조건은 없다.",
        }
    )
    trust_with_low_debt = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "수도권",
            "deposit_million": 160,
            "estimated_market_price_million": 330,
            "trust_registered": True,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "broker_explained_rights": False,
            "special_clause_text": "신탁원부와 수탁자 동의서는 계약 후 전달한다고 되어 있다.",
        }
    )
    high_ratio_only = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "비수도권",
            "deposit_million": 205,
            "estimated_market_price_million": 250,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "nearby_market_gap_percent": 8,
        }
    )
    assert seizure_with_low_ratio["risk_grade"] == "위험"
    assert trust_with_low_debt["risk_score"] >= 72
    assert high_ratio_only["risk_grade"] == "주의"


def test_emerging_text_risk_patterns_are_handled():
    scorer = RiskScorer()
    identity_mismatch = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 190,
            "estimated_market_price_million": 330,
            "broker_advertising_issue": True,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "broker_explained_rights": False,
            "special_clause_text": "대리인이 계약하고 위임장 미확인 상태이며 인감증명 미확인이다.",
            "user_situation_text": "등기부 소유자와 계약 진행자 신분증 명의 불일치가 있다.",
        }
    )
    delayed_move_in = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "광역시",
            "deposit_million": 260,
            "estimated_market_price_million": 390,
            "mortgage_million": 70,
            "suspicious_special_clause": True,
            "guarantee_insurance_available": False,
            "fixed_date_ready": False,
            "move_in_ready": False,
            "broker_explained_rights": False,
            "special_clause_text": "잔금 후 전입신고 지연 요구가 있고 잔금 후 담보대출 가능성이 있다.",
            "user_situation_text": "전입 전 근저당 설정으로 대항력 공백이 생길 수 있다.",
        }
    )
    verified_proxy = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 230,
            "estimated_market_price_million": 540,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "special_clause_text": "위임장 원본, 인감증명, 소유자 영상통화 확인을 계약서에 첨부했다.",
            "user_situation_text": "대리계약이지만 본인 의사와 권리관계를 모두 확인했다.",
        }
    )
    assert identity_mismatch["risk_score"] >= 74
    assert identity_mismatch["risk_grade"] == "위험"
    assert delayed_move_in["risk_score"] >= 76
    assert delayed_move_in["risk_grade"] == "위험"
    assert verified_proxy["risk_score"] < 45


def test_colloquial_text_only_inputs_are_not_missed():
    scorer = RiskScorer()
    text_only_seizure = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 170,
            "estimated_market_price_million": 430,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "등기부에 압류가 있다는데 집주인이 곧 풀린다더라 그냥 계약해도 된다네요.",
        }
    )
    text_only_trust = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "officetel",
            "region": "수도권",
            "deposit_million": 180,
            "estimated_market_price_million": 370,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "신탁 표시가 있는데 신탁원부 없어도 된대요. 수탁자 동의는 나중에 받자고 합니다.",
        }
    )
    colloquial_safe = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 220,
            "estimated_market_price_million": 560,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "등기부 깨끗하고 갑구 을구 깨끗, 보증보험 가능, 확정일자랑 전입 바로 가능해요.",
        }
    )
    assert text_only_seizure["risk_grade"] == "위험"
    assert text_only_seizure["risk_score"] >= 72
    assert text_only_trust["risk_grade"] == "위험"
    assert text_only_trust["risk_score"] >= 72
    assert colloquial_safe["risk_score"] < 45
    assert colloquial_safe["risk_grade"] == "안전"


def test_payment_account_and_building_text_only_inputs_are_not_missed():
    scorer = RiskScorer()
    account_mismatch = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 190,
            "estimated_market_price_million": 360,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "계약금은 집주인 말고 다른 사람 계좌로 보내래요. 계좌 명의가 임대인 이름이 아닙니다.",
        }
    )
    illegal_split = scorer.score(
        {
            "contract_type": "monthly_rent",
            "property_type": "multi_family",
            "region": "수도권",
            "deposit_million": 70,
            "monthly_rent_million": 0.7,
            "estimated_market_price_million": 240,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "원룸이 불법증축 쪼개기 방일 수 있다는데 건축물대장은 아직 못 봤습니다.",
        }
    )
    pressure_deposit = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 230,
            "estimated_market_price_million": 310,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "오늘 계약금 먼저 넣어야 잡아준다고 하고 등기부는 나중에 보자고 합니다.",
        }
    )
    safe_verified_payment = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 220,
            "estimated_market_price_million": 560,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "임대인 계좌 일치 소유자 계좌 확인 완료 계약금 보류 등기부 확인 완료 건축물대장 확인 완료 건축물대장 정상입니다.",
        }
    )
    assert account_mismatch["risk_grade"] == "위험"
    assert account_mismatch["risk_score"] >= 74
    assert illegal_split["risk_grade"] == "위험"
    assert illegal_split["risk_score"] >= 70
    assert pressure_deposit["risk_grade"] in {"주의", "위험"}
    assert pressure_deposit["risk_score"] >= 64
    assert safe_verified_payment["risk_grade"] == "안전"
    assert safe_verified_payment["risk_score"] < 45


def test_tenancy_title_text_only_inputs_are_not_missed():
    scorer = RiskScorer()
    unauthorized_sublease = scorer.score(
        {
            "contract_type": "monthly_rent",
            "property_type": "officetel",
            "region": "수도권",
            "deposit_million": 50,
            "monthly_rent_million": 0.8,
            "estimated_market_price_million": 240,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "집주인이 아니라 기존 세입자가 전대차로 계약하자고 합니다. 임대인 동의서는 없고 전입신고는 가능하다고만 해요.",
        }
    )
    lease_registration = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "apartment",
            "region": "수도권",
            "deposit_million": 250,
            "estimated_market_price_million": 520,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "등기부에 임차권등기명령이 남아 있는데 전 세입자와는 해결됐다고만 합니다.",
        }
    )
    unregistered_new_build = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 230,
            "estimated_market_price_million": 330,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "신축인데 아직 미등기이고 사용승인 전이라 등기부등본이 없다고 합니다.",
        }
    )
    non_refundable_reservation = scorer.score(
        {
            "contract_type": "jeonse",
            "property_type": "villa",
            "region": "수도권",
            "deposit_million": 210,
            "estimated_market_price_million": 330,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "가계약금 넣으면 환불불가라고 하고 계약서 보기 전 먼저 송금하라고 합니다.",
        }
    )
    safe_sublease_like = scorer.score(
        {
            "contract_type": "monthly_rent",
            "property_type": "officetel",
            "region": "수도권",
            "deposit_million": 50,
            "monthly_rent_million": 0.8,
            "estimated_market_price_million": 240,
            "guarantee_insurance_available": True,
            "fixed_date_ready": True,
            "move_in_ready": True,
            "broker_explained_rights": True,
            "user_situation_text": "전대차가 아니라 임대인 본인과 직접 계약하고 임대인 동의서 확인 완료했습니다.",
        }
    )
    assert unauthorized_sublease["risk_grade"] == "위험"
    assert unauthorized_sublease["risk_score"] >= 70
    assert lease_registration["risk_grade"] == "위험"
    assert lease_registration["risk_score"] >= 72
    assert unregistered_new_build["risk_grade"] == "위험"
    assert unregistered_new_build["risk_score"] >= 74
    assert non_refundable_reservation["risk_grade"] in {"주의", "위험"}
    assert non_refundable_reservation["risk_score"] >= 64
    assert safe_sublease_like["risk_grade"] == "안전"
    assert safe_sublease_like["risk_score"] < 45
