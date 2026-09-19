"""Generate synthetic 0/1-knapsack instances."""

import random
from dataclasses import dataclass

from .models import Item


@dataclass(frozen=True)
class KnapsackInstance:
    """One generated instance together with its generation metadata."""

    items: tuple[Item, ...]
    capacity: int
    seed: int
    family: str = "independent"


def generate_independent_instance(
    number_of_items: int,
    seed: int,
    *,
    weight_range: tuple[int, int] = (1, 50),
    value_range: tuple[int, int] = (1, 100),
    capacity_ratio: float = 0.5,
) -> tuple[tuple[Item, ...], int]:
    """Generate one instance with independently sampled weights and values.

    Both ranges are inclusive. The capacity is ``capacity_ratio`` times the
    total item weight, rounded to the nearest integer. Supplying a seed makes
    an instance reproducible and will later let us split datasets by seed.
    """
    if number_of_items <= 0:
        raise ValueError("number_of_items must be positive")

    minimum_weight, maximum_weight = weight_range
    minimum_value, maximum_value = value_range
    if minimum_weight <= 0 or minimum_weight > maximum_weight:
        raise ValueError("weight_range must contain positive integers in order")
    if minimum_value <= 0 or minimum_value > maximum_value:
        raise ValueError("value_range must contain positive integers in order")
    if not 0 < capacity_ratio < 1:
        raise ValueError("capacity_ratio must be between 0 and 1")

    random_number_generator = random.Random(seed)
    items = tuple(
        Item(
            name=f"item_{index}",
            weight=random_number_generator.randint(minimum_weight, maximum_weight),
            value=random_number_generator.randint(minimum_value, maximum_value),
        )
        for index in range(number_of_items)
    )
    capacity = round(sum(item.weight for item in items) * capacity_ratio)

    return items, capacity


def generate_independent_instances(
    number_of_instances: int = 1_000,
    *,
    number_of_items: int = 20,
    starting_seed: int = 0,
    weight_range: tuple[int, int] = (1, 50),
    value_range: tuple[int, int] = (1, 100),
    capacity_ratio: float = 0.5,
) -> tuple[KnapsackInstance, ...]:
    """Generate a reproducible batch of independent instances.

    Consecutive seeds beginning at ``starting_seed`` give every instance a
    stable identity that can later be used for train/test splitting.
    """
    if number_of_instances <= 0:
        raise ValueError("number_of_instances must be positive")

    instances = []
    for seed in range(starting_seed, starting_seed + number_of_instances):
        items, capacity = generate_independent_instance(
            number_of_items,
            seed,
            weight_range=weight_range,
            value_range=value_range,
            capacity_ratio=capacity_ratio,
        )
        instances.append(
            KnapsackInstance(items=items, capacity=capacity, seed=seed)
        )

    return tuple(instances)
