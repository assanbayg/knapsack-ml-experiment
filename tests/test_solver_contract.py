import unittest
from collections import Counter

from knapsack_ml_experiment import (
    Item,
    solve_dynamic_programming,
    solve_greedy,
)


class TestSolverContract(unittest.TestCase):
    """Check properties that every knapsack solver must satisfy."""

    def setUp(self) -> None:
        self.solvers = (solve_greedy, solve_dynamic_programming)

    def test_solutions_are_feasible_and_totals_are_consistent(self) -> None:
        cases = [
            (
                [
                    Item("A", weight=2, value=6),
                    Item("B", weight=3, value=8),
                    Item("C", weight=4, value=9),
                ],
                7,
            ),
            (
                [
                    Item("A", weight=10, value=60),
                    Item("B", weight=20, value=100),
                    Item("C", weight=30, value=120),
                ],
                50,
            ),
            ([Item("too-heavy", weight=11, value=100)], 10),
            ([], 10),
        ]

        for solver in self.solvers:
            for items, capacity in cases:
                with self.subTest(solver=solver.__name__, capacity=capacity):
                    solution = solver(items, capacity)

                    self.assertLessEqual(solution.total_weight, capacity)
                    self.assertEqual(
                        solution.total_weight,
                        sum(item.weight for item in solution.selected_items),
                    )
                    self.assertEqual(
                        solution.total_value,
                        sum(item.value for item in solution.selected_items),
                    )
                    self.assertFalse(
                        Counter(solution.selected_items) - Counter(items)
                    )

    def test_solvers_agree_when_greedy_is_optimal(self) -> None:
        items = [
            Item("A", weight=1, value=4),
            Item("B", weight=2, value=6),
            Item("C", weight=3, value=6),
        ]

        greedy_solution = solve_greedy(items, capacity=3)
        dp_solution = solve_dynamic_programming(items, capacity=3)

        self.assertEqual(greedy_solution.total_value, 10)
        self.assertEqual(dp_solution.total_value, 10)

    def test_dp_beats_greedy_on_classic_counterexample(self) -> None:
        items = [
            Item("A", weight=10, value=60),
            Item("B", weight=20, value=100),
            Item("C", weight=30, value=120),
        ]

        greedy_solution = solve_greedy(items, capacity=50)
        dp_solution = solve_dynamic_programming(items, capacity=50)

        self.assertEqual(greedy_solution.total_value, 160)
        self.assertEqual(dp_solution.total_value, 220)


if __name__ == "__main__":
    unittest.main()
