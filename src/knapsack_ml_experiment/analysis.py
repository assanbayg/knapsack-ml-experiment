"""Label generated instances and visualize greedy failures."""

import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from matplotlib.figure import Figure

from .dynamic_programming import solve_dynamic_programming
from .generation import KnapsackInstance
from .greedy import solve_greedy


def evaluate_instance(instance: KnapsackInstance) -> dict[str, object]:
    """Compare greedy with the exact optimum for one generated instance."""
    greedy_solution = solve_greedy(instance.items, instance.capacity)
    optimal_solution = solve_dynamic_programming(instance.items, instance.capacity)

    greedy_value = greedy_solution.total_value
    optimal_value = optimal_solution.total_value
    failure = greedy_value < optimal_value
    relative_gap = (
        (optimal_value - greedy_value) / optimal_value
        if optimal_value > 0
        else 0.0
    )

    return {
        "seed": instance.seed,
        "family": instance.family,
        "number_of_items": len(instance.items),
        "capacity": instance.capacity,
        "weights": tuple(item.weight for item in instance.items),
        "values": tuple(item.value for item in instance.items),
        "greedy_value": greedy_value,
        "optimal_value": optimal_value,
        "failure": failure,
        "relative_gap": relative_gap,
    }


def build_labeled_dataset(
    instances: Iterable[KnapsackInstance],
) -> pd.DataFrame:
    """Create one labeled row per generated knapsack instance."""
    rows = [evaluate_instance(instance) for instance in instances]
    return pd.DataFrame(rows)


def split_dataset_by_seed(
    dataset: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows while keeping every occurrence of a seed together.

    A seed may occur once per instance family. Assigning unique seeds rather
    than individual rows prevents the same seed from appearing in multiple
    splits and preserves family balance when families share the same seeds.
    """
    if "seed" not in dataset.columns:
        raise ValueError("dataset must contain a seed column")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    seeds = dataset["seed"].drop_duplicates().tolist()
    random.Random(random_seed).shuffle(seeds)

    train_count = round(len(seeds) * train_fraction)
    validation_count = round(len(seeds) * validation_fraction)
    test_count = len(seeds) - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("dataset does not contain enough unique seeds for this split")

    train_seeds = set(seeds[:train_count])
    validation_seeds = set(
        seeds[train_count : train_count + validation_count]
    )
    test_seeds = set(seeds[train_count + validation_count :])

    def rows_for(selected_seeds: set[object]) -> pd.DataFrame:
        return dataset.loc[dataset["seed"].isin(selected_seeds)].reset_index(
            drop=True
        )

    return (
        rows_for(train_seeds),
        rows_for(validation_seeds),
        rows_for(test_seeds),
    )


def save_failure_plots(dataset: pd.DataFrame, output_directory: Path) -> None:
    """Save failure-frequency and relative-gap plots as PNG files."""
    required_columns = {"failure", "relative_gap"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing required columns: {missing}")

    output_directory.mkdir(parents=True, exist_ok=True)

    failure_counts = dataset["failure"].value_counts().reindex(
        [False, True], fill_value=0
    )
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.bar(["Greedy optimal", "Greedy failed"], failure_counts.values)
    axis.set_ylabel("Number of instances")
    axis.set_title("Ratio-greedy outcomes")
    figure.tight_layout()
    figure.savefig(output_directory / "failure_frequency.png", dpi=150)

    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.hist(dataset.loc[dataset["failure"], "relative_gap"], bins=20)
    axis.set_xlabel("Relative optimality gap")
    axis.set_ylabel("Number of failed instances")
    axis.set_title("Gap distribution when ratio-greedy fails")
    figure.tight_layout()
    figure.savefig(output_directory / "relative_gap_distribution.png", dpi=150)
