#!/usr/bin/env python3
from __future__ import annotations

import json

from risk_detector.model.training import train_model


if __name__ == "__main__":
    metadata = train_model(rebuild_data=True)
    print(json.dumps({"model_path": metadata["model_path"], "metrics": metadata["metrics"]}, ensure_ascii=False, indent=2))

