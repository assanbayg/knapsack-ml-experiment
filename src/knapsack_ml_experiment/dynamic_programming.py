"""Exact dynamic-programming solver for 0/1 knapsack."""

from collections.abc import Iterable

from .models import Item, KnapsackSolution


def solve_dynamic_programming(
    items: Iterable[Item], capacity: int
) -> KnapsackSolution:
    """Find an optimal 0/1 knapsack solution using a 2D DP table.

    ``table[i][current_capacity]`` stores the greatest value obtainable using
    only the first ``i`` items. After filling the table, the selected items are
    reconstructed by walking backward through it.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    item_list = list(items)
    number_of_items = len(item_list)
    table = [
        [0] * (capacity + 1)
        for _ in range(number_of_items + 1)
    ]

    for item_count in range(1, number_of_items + 1):
        item = item_list[item_count - 1]

        for current_capacity in range(capacity + 1):
            value_without_item = table[item_count - 1][current_capacity]

            if item.weight > current_capacity:
                table[item_count][current_capacity] = value_without_item
                continue

            value_with_item = (
                item.value
                + table[item_count - 1][current_capacity - item.weight]
            )
            table[item_count][current_capacity] = max(
                value_without_item,
                value_with_item,
            )

    selected_items: list[Item] = []
    remaining_capacity = capacity

    for item_count in range(number_of_items, 0, -1):
        value_with_first_items = table[item_count][remaining_capacity]
        value_without_last_item = table[item_count - 1][remaining_capacity]

        if value_with_first_items == value_without_last_item:
            continue

        item = item_list[item_count - 1]
        selected_items.append(item)
        remaining_capacity -= item.weight

    selected_items.reverse()

    return KnapsackSolution(
        selected_items=tuple(selected_items),
        total_weight=sum(item.weight for item in selected_items),
        total_value=table[number_of_items][capacity],
    )
