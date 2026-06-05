#!/usr/bin/env python3
from __future__ import annotations

import json

from risk_detector.data.pipeline import build_all_datasets


if __name__ == "__main__":
    print(json.dumps(build_all_datasets(), ensure_ascii=False, indent=2))

