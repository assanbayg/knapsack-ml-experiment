"""Ratio-based greedy solver for 0/1 knapsack."""

from collections.abc import Iterable

from .models import Item, KnapsackSolution


def solve_greedy(items: Iterable[Item], capacity: int) -> KnapsackSolution:
    """Select items in descending value-to-weight ratio order.

    The function considers each item once. An item is selected if it fits in
    the remaining capacity; otherwise it is skipped. Because this is 0/1
    knapsack, an item is either selected whole or not selected at all.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    ranked_items = sorted(
        items,
        key=lambda item: item.value_to_weight_ratio,
        reverse=True,
    )

    selected_items: list[Item] = []
    total_weight = 0
    total_value = 0

    for item in ranked_items:
        if total_weight + item.weight <= capacity:
            selected_items.append(item)
            total_weight += item.weight
            total_value += item.value

    return KnapsackSolution(
        selected_items=tuple(selected_items),
        total_weight=total_weight,
        total_value=total_value,
    )
