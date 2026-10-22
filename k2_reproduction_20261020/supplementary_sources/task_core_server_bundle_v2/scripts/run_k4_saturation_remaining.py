from __future__ import annotations
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
STAGE = "K4_SATURATION"

def main() -> None:
    cases = json.loads((BUNDLE / "manifests" / "SERVER_REQUIRED_CASES.json").read_text(encoding="utf-8"))["cases"]
    pending = [row for row in cases if row["experiment"] == STAGE]
    if pending:
        raise RuntimeError(f"UNEXPECTED_UNIMPLEMENTED_SERVER_CASES:{STAGE}:{len(pending)}")
    print(f"{STAGE}: no server cases; local result is complete or R1 is structurally inapplicable")

if __name__ == "__main__":
    main()
