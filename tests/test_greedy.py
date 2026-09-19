import unittest

from knapsack_ml_experiment import Item, solve_greedy


class TestGreedySolver(unittest.TestCase):
    def test_returns_empty_solution_for_empty_items(self) -> None:
        solution = solve_greedy([], capacity=10)

        self.assertEqual(solution.selected_items, ())
        self.assertEqual(solution.total_weight, 0)
        self.assertEqual(solution.total_value, 0)

    def test_selects_items_by_ratio_when_they_fit(self) -> None:
        items = [
            Item("A", weight=2, value=6),
            Item("B", weight=3, value=8),
            Item("C", weight=4, value=9),
        ]

        solution = solve_greedy(items, capacity=7)

        self.assertEqual([item.name for item in solution.selected_items], ["A", "B"])
        self.assertEqual(solution.total_weight, 5)
        self.assertEqual(solution.total_value, 14)

    def test_skips_an_item_that_does_not_fit(self) -> None:
        items = [
            Item("heavy", weight=8, value=80),
            Item("small", weight=3, value=15),
        ]

        solution = solve_greedy(items, capacity=5)

        self.assertEqual(solution.selected_items, (items[1],))

    def test_zero_capacity_selects_nothing(self) -> None:
        solution = solve_greedy([Item("A", weight=1, value=1)], capacity=0)

        self.assertEqual(solution.selected_items, ())
        self.assertEqual(solution.total_weight, 0)
        self.assertEqual(solution.total_value, 0)

    def test_negative_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_greedy([], capacity=-1)


if __name__ == "__main__":
    unittest.main()
