from .dynamic_programming import solve_dynamic_programming
from .greedy import solve_greedy
from .models import Item, KnapsackSolution

__all__ = [
    "Item",
    "KnapsackSolution",
    "solve_dynamic_programming",
    "solve_greedy",
]


def main() -> None:
    """Run a small demonstration of the greedy solver."""
    items = [
        Item("A", weight=2, value=6),
        Item("B", weight=3, value=8),
        Item("C", weight=4, value=9),
    ]
    solution = solve_greedy(items, capacity=7)

    names = ", ".join(item.name for item in solution.selected_items)
    print(f"Selected items: {names}")
    print(f"Total weight: {solution.total_weight}")
    print(f"Total value: {solution.total_value}")
