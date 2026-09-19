"""Data structures shared by the knapsack solvers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """One item that may be placed in a 0/1 knapsack."""

    name: str
    weight: int
    value: int

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("item weight must be positive")
        if self.value < 0:
            raise ValueError("item value must be non-negative")

    @property
    def value_to_weight_ratio(self) -> float:
        return self.value / self.weight


@dataclass(frozen=True)
class KnapsackSolution:
    """The items selected by a solver and their combined totals."""

    selected_items: tuple[Item, ...]
    total_weight: int
    total_value: int
