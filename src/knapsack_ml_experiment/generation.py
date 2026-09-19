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


def _validate_common_parameters(
    number_of_items: int,
    weight_range: tuple[int, int],
    capacity_ratio: float,
) -> None:
    """Validate parameters shared by all stochastic generators."""
    if number_of_items <= 0:
        raise ValueError("number_of_items must be positive")

    minimum_weight, maximum_weight = weight_range
    if minimum_weight <= 0 or minimum_weight > maximum_weight:
        raise ValueError("weight_range must contain positive integers in order")
    if not 0 < capacity_ratio < 1:
        raise ValueError("capacity_ratio must be between 0 and 1")


def _capacity_from_items(
    items: tuple[Item, ...], capacity_ratio: float
) -> int:
    return round(sum(item.weight for item in items) * capacity_ratio)


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
    _validate_common_parameters(number_of_items, weight_range, capacity_ratio)

    minimum_weight, maximum_weight = weight_range
    minimum_value, maximum_value = value_range
    if minimum_value <= 0 or minimum_value > maximum_value:
        raise ValueError("value_range must contain positive integers in order")

    random_number_generator = random.Random(seed)
    items = tuple(
        Item(
            name=f"item_{index}",
            weight=random_number_generator.randint(minimum_weight, maximum_weight),
            value=random_number_generator.randint(minimum_value, maximum_value),
        )
        for index in range(number_of_items)
    )
    capacity = _capacity_from_items(items, capacity_ratio)

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


def generate_strongly_correlated_instance(
    number_of_items: int,
    seed: int,
    *,
    weight_range: tuple[int, int] = (1, 50),
    value_offset: int = 10,
    capacity_ratio: float = 0.5,
) -> tuple[tuple[Item, ...], int]:
    """Generate an instance where every value equals weight plus an offset."""
    _validate_common_parameters(number_of_items, weight_range, capacity_ratio)
    if value_offset < 0:
        raise ValueError("value_offset must be non-negative")

    random_number_generator = random.Random(seed)
    items = tuple(
        Item(
            name=f"item_{index}",
            weight=(weight := random_number_generator.randint(*weight_range)),
            value=weight + value_offset,
        )
        for index in range(number_of_items)
    )
    return items, _capacity_from_items(items, capacity_ratio)


def generate_weakly_correlated_instance(
    number_of_items: int,
    seed: int,
    *,
    weight_range: tuple[int, int] = (1, 50),
    value_spread: int = 10,
    capacity_ratio: float = 0.5,
) -> tuple[tuple[Item, ...], int]:
    """Generate values uniformly near their corresponding item weights."""
    _validate_common_parameters(number_of_items, weight_range, capacity_ratio)
    if value_spread < 0:
        raise ValueError("value_spread must be non-negative")

    random_number_generator = random.Random(seed)
    items = []
    for index in range(number_of_items):
        weight = random_number_generator.randint(*weight_range)
        value = random_number_generator.randint(
            max(1, weight - value_spread), weight + value_spread
        )
        items.append(Item(name=f"item_{index}", weight=weight, value=value))

    result = tuple(items)
    return result, _capacity_from_items(result, capacity_ratio)


def generate_similar_ratio_instance(
    number_of_items: int,
    seed: int,
    *,
    weight_range: tuple[int, int] = (1, 50),
    target_ratio: float = 2.0,
    ratio_jitter: float = 0.05,
    capacity_ratio: float = 0.5,
) -> tuple[tuple[Item, ...], int]:
    """Generate items whose value-to-weight ratios lie in a narrow band."""
    _validate_common_parameters(number_of_items, weight_range, capacity_ratio)
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    if not 0 <= ratio_jitter < 1:
        raise ValueError("ratio_jitter must be between 0 (inclusive) and 1")

    random_number_generator = random.Random(seed)
    items = []
    for index in range(number_of_items):
        weight = random_number_generator.randint(*weight_range)
        ratio = target_ratio * random_number_generator.uniform(
            1 - ratio_jitter, 1 + ratio_jitter
        )
        items.append(
            Item(
                name=f"item_{index}",
                weight=weight,
                value=max(1, round(weight * ratio)),
            )
        )

    result = tuple(items)
    return result, _capacity_from_items(result, capacity_ratio)


def _generate_batch(
    generator,
    family: str,
    number_of_instances: int,
    number_of_items: int,
    starting_seed: int,
    **generator_arguments,
) -> tuple[KnapsackInstance, ...]:
    if number_of_instances <= 0:
        raise ValueError("number_of_instances must be positive")

    instances = []
    for seed in range(starting_seed, starting_seed + number_of_instances):
        items, capacity = generator(
            number_of_items, seed, **generator_arguments
        )
        instances.append(
            KnapsackInstance(items, capacity, seed, family=family)
        )
    return tuple(instances)


def generate_strongly_correlated_instances(
    number_of_instances: int = 1_000,
    *,
    number_of_items: int = 20,
    starting_seed: int = 0,
    **generator_arguments,
) -> tuple[KnapsackInstance, ...]:
    """Generate a reproducible batch of strongly correlated instances."""
    return _generate_batch(
        generate_strongly_correlated_instance,
        "strongly_correlated",
        number_of_instances,
        number_of_items,
        starting_seed,
        **generator_arguments,
    )


def generate_weakly_correlated_instances(
    number_of_instances: int = 1_000,
    *,
    number_of_items: int = 20,
    starting_seed: int = 0,
    **generator_arguments,
) -> tuple[KnapsackInstance, ...]:
    """Generate a reproducible batch of weakly correlated instances."""
    return _generate_batch(
        generate_weakly_correlated_instance,
        "weakly_correlated",
        number_of_instances,
        number_of_items,
        starting_seed,
        **generator_arguments,
    )


def generate_similar_ratio_instances(
    number_of_instances: int = 1_000,
    *,
    number_of_items: int = 20,
    starting_seed: int = 0,
    **generator_arguments,
) -> tuple[KnapsackInstance, ...]:
    """Generate a reproducible batch of similar-ratio instances."""
    return _generate_batch(
        generate_similar_ratio_instance,
        "similar_ratio",
        number_of_instances,
        number_of_items,
        starting_seed,
        **generator_arguments,
    )
