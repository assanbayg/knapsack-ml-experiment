"""Hybrid solver that routes likely greedy failures to exact DP."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dynamic_programming import solve_dynamic_programming
from .greedy import solve_greedy
from .models import Item, KnapsackSolution


FEATURE_COLUMNS = (
    "number_of_items",
    "capacity_ratio",
    "weight_mean",
    "weight_std",
    "value_mean",
    "value_std",
    "ratio_mean",
    "ratio_std",
    "weight_value_correlation",
)


@dataclass(frozen=True)
class HybridResult:
    """A hybrid solution together with the reason for its routing choice."""

    solution: KnapsackSolution
    solver_used: str
    failure_probability: float


def extract_hybrid_features(
    items: Iterable[Item], capacity: int
) -> pd.DataFrame:
    """Create the same non-leaking feature row used to train the classifier."""
    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    item_list = list(items)
    if not item_list:
        raise ValueError("the hybrid classifier requires at least one item")

    weights = np.array([item.weight for item in item_list], dtype=float)
    values = np.array([item.value for item in item_list], dtype=float)
    ratios = values / weights

    correlation = 0.0
    if len(item_list) > 1 and weights.std() > 0 and values.std() > 0:
        correlation = float(np.corrcoef(weights, values)[0, 1])

    feature_values = {
        "number_of_items": len(item_list),
        "capacity_ratio": capacity / weights.sum(),
        "weight_mean": weights.mean(),
        "weight_std": weights.std(),
        "value_mean": values.mean(),
        "value_std": values.std(),
        "ratio_mean": ratios.mean(),
        "ratio_std": ratios.std(),
        "weight_value_correlation": correlation,
    }
    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)


def solve_hybrid(
    items: Iterable[Item],
    capacity: int,
    classifier: Any,
    threshold: float = 0.5,
) -> HybridResult:
    """Use DP above the failure threshold; otherwise accept greedy."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    item_list = list(items)
    features = extract_hybrid_features(item_list, capacity)
    probabilities = classifier.predict_proba(features)[0]

    classes = list(classifier.classes_)
    try:
        failure_index = classes.index(1)
    except ValueError as error:
        raise ValueError("classifier must have a failure class labelled 1") from error

    failure_probability = float(probabilities[failure_index])
    if failure_probability > threshold:
        solution = solve_dynamic_programming(item_list, capacity)
        solver_used = "dynamic_programming"
    else:
        solution = solve_greedy(item_list, capacity)
        solver_used = "greedy"

    return HybridResult(
        solution=solution,
        solver_used=solver_used,
        failure_probability=failure_probability,
    )
