import unittest

from knapsack_ml_experiment import Item, solve_dynamic_programming


class TestDynamicProgrammingSolver(unittest.TestCase):
    def test_finds_optimum_when_ratio_greedy_fails(self) -> None:
        items = [
            Item("A", weight=2, value=6),
            Item("B", weight=3, value=8),
            Item("C", weight=4, value=9),
        ]

        solution = solve_dynamic_programming(items, capacity=7)

        self.assertEqual(
            [item.name for item in solution.selected_items],
            ["B", "C"],
        )
        self.assertEqual(solution.total_weight, 7)
        self.assertEqual(solution.total_value, 17)

    def test_returns_empty_solution_for_empty_items(self) -> None:
        solution = solve_dynamic_programming([], capacity=10)

        self.assertEqual(solution.selected_items, ())
        self.assertEqual(solution.total_weight, 0)
        self.assertEqual(solution.total_value, 0)

    def test_skips_items_that_are_too_heavy(self) -> None:
        items = [Item("heavy", weight=6, value=100)]

        solution = solve_dynamic_programming(items, capacity=5)

        self.assertEqual(solution.selected_items, ())
        self.assertEqual(solution.total_value, 0)

    def test_zero_capacity_selects_nothing(self) -> None:
        solution = solve_dynamic_programming(
            [Item("A", weight=1, value=1)],
            capacity=0,
        )

        self.assertEqual(solution.selected_items, ())
        self.assertEqual(solution.total_weight, 0)
        self.assertEqual(solution.total_value, 0)

    def test_negative_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_dynamic_programming([], capacity=-1)


if __name__ == "__main__":
    unittest.main()
