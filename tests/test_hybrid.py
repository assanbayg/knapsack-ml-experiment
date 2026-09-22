import unittest

import numpy as np

from knapsack_ml_experiment import Item, solve_hybrid


class StubClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, failure_probability: float) -> None:
        self.failure_probability = failure_probability
        self.received_columns: list[str] = []

    def predict_proba(self, features):
        self.received_columns = features.columns.tolist()
        return np.array(
            [[1 - self.failure_probability, self.failure_probability]]
        )


class TestHybridSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [
            Item("A", weight=10, value=60),
            Item("B", weight=20, value=100),
            Item("C", weight=30, value=120),
        ]

    def test_runs_dp_when_failure_probability_exceeds_threshold(self) -> None:
        result = solve_hybrid(
            self.items,
            capacity=50,
            classifier=StubClassifier(0.8),
            threshold=0.5,
        )

        self.assertEqual(result.solver_used, "dynamic_programming")
        self.assertEqual(result.solution.total_value, 220)
        self.assertEqual(result.failure_probability, 0.8)

    def test_accepts_greedy_when_probability_does_not_exceed_threshold(self) -> None:
        result = solve_hybrid(
            self.items,
            capacity=50,
            classifier=StubClassifier(0.5),
            threshold=0.5,
        )

        self.assertEqual(result.solver_used, "greedy")
        self.assertEqual(result.solution.total_value, 160)

    def test_rejects_threshold_outside_probability_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            solve_hybrid(
                self.items,
                capacity=50,
                classifier=StubClassifier(0.5),
                threshold=1.1,
            )

    def test_rejects_an_empty_instance(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one item"):
            solve_hybrid(
                [],
                capacity=50,
                classifier=StubClassifier(0.5),
            )


if __name__ == "__main__":
    unittest.main()
