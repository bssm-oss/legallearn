#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from risk_detector.risk.scorer import RiskScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="부동산 계약 위험도 예측")
    parser.add_argument("--json", required=True, help="계약 정보 JSON 문자열")
    args = parser.parse_args()
    payload = json.loads(args.json)
    result = RiskScorer().score(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

