from .analysis import build_labeled_dataset, evaluate_instance, save_failure_plots
from .dynamic_programming import solve_dynamic_programming
from .generation import (
    KnapsackInstance,
    generate_independent_instance,
    generate_independent_instances,
    generate_similar_ratio_instance,
    generate_similar_ratio_instances,
    generate_strongly_correlated_instance,
    generate_strongly_correlated_instances,
    generate_weakly_correlated_instance,
    generate_weakly_correlated_instances,
)
from .greedy import solve_greedy
from .models import Item, KnapsackSolution

__all__ = [
    "Item",
    "KnapsackInstance",
    "KnapsackSolution",
    "build_labeled_dataset",
    "evaluate_instance",
    "generate_independent_instance",
    "generate_independent_instances",
    "generate_similar_ratio_instance",
    "generate_similar_ratio_instances",
    "generate_strongly_correlated_instance",
    "generate_strongly_correlated_instances",
    "generate_weakly_correlated_instance",
    "generate_weakly_correlated_instances",
    "save_failure_plots",
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
