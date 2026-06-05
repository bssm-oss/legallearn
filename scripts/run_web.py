#!/usr/bin/env python3
from __future__ import annotations

import argparse

from risk_detector.web.app import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="로컬 웹 데모 실행")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run(host=args.host, port=args.port)

