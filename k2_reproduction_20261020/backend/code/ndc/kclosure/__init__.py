"""Dimension-generic simplicial closure utilities.

The legacy k=2 entry points remain unchanged.  The k=3 smoke runner imports
this package explicitly so candidate caches and model artifacts cannot collide
with the historical triangle-closure pipeline.
"""

from .core import (
    build_candidate_task,
    build_derived_graph,
    build_observed_state,
    enumerate_open_simplices,
    load_timestamped_simplices,
)

__all__ = [
    "build_candidate_task",
    "build_derived_graph",
    "build_observed_state",
    "enumerate_open_simplices",
    "load_timestamped_simplices",
]
