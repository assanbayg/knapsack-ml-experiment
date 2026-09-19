import tempfile
import unittest
from pathlib import Path

from knapsack_ml_experiment import (
    Item,
    KnapsackInstance,
    build_labeled_dataset,
    evaluate_instance,
    save_failure_plots,
)


class TestInstanceEvaluation(unittest.TestCase):
    def test_records_failure_and_relative_gap(self) -> None:
        instance = KnapsackInstance(
            items=(
                Item("A", weight=10, value=60),
                Item("B", weight=20, value=100),
                Item("C", weight=30, value=120),
            ),
            capacity=50,
            seed=42,
        )

        row = evaluate_instance(instance)

        self.assertEqual(row["greedy_value"], 160)
        self.assertEqual(row["optimal_value"], 220)
        self.assertIs(row["failure"], True)
        self.assertAlmostEqual(row["relative_gap"], 60 / 220)
        self.assertEqual(row["family"], "independent")
        self.assertEqual(row["seed"], 42)

    def test_records_zero_gap_when_greedy_is_optimal(self) -> None:
        instance = KnapsackInstance(
            items=(Item("only", weight=1, value=5),),
            capacity=1,
            seed=1,
        )

        row = evaluate_instance(instance)

        self.assertIs(row["failure"], False)
        self.assertEqual(row["relative_gap"], 0.0)

    def test_builds_one_dataset_row_per_instance(self) -> None:
        instances = (
            KnapsackInstance((Item("A", 1, 2),), 1, seed=10),
            KnapsackInstance((Item("B", 2, 3),), 2, seed=11),
        )

        dataset = build_labeled_dataset(instances)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset["seed"].tolist(), [10, 11])
        self.assertIn("greedy_value", dataset.columns)
        self.assertIn("optimal_value", dataset.columns)
        self.assertIn("failure", dataset.columns)
        self.assertIn("relative_gap", dataset.columns)


class TestFailurePlots(unittest.TestCase):
    def test_saves_both_requested_plots(self) -> None:
        instances = (
            KnapsackInstance((Item("A", 1, 2),), 1, seed=10),
            KnapsackInstance((Item("B", 2, 3),), 2, seed=11),
        )
        dataset = build_labeled_dataset(instances)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            save_failure_plots(dataset, output_directory)

            self.assertTrue((output_directory / "failure_frequency.png").exists())
            self.assertTrue(
                (output_directory / "relative_gap_distribution.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
