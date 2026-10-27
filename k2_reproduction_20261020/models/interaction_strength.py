from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EdgeStrength:
    count: float
    log_count: float
    decay: float
    recency: float
    last_time: float

def strength_value(record: EdgeStrength | None, mode: str) -> float:
    if mode == "binary":
        return 1.0
    if record is None:
        return 0.0
    if mode in {"weighted-count", "boundary-count"}:
        return float(record.count)
    if mode in {"weighted-log", "boundary-log"}:
        return float(record.log_count)
    if mode in {"weighted-decay", "boundary-decay"}:
        return float(record.decay)
    raise ValueError(f"Unknown strength mode: {mode}")
