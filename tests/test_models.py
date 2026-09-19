import unittest

from knapsack_ml_experiment import Item


class TestItem(unittest.TestCase):
    def test_calculates_value_to_weight_ratio(self) -> None:
        item = Item("A", weight=4, value=10)

        self.assertEqual(item.value_to_weight_ratio, 2.5)

    def test_rejects_zero_weight(self) -> None:
        with self.assertRaises(ValueError):
            Item("A", weight=0, value=10)

    def test_rejects_negative_weight(self) -> None:
        with self.assertRaises(ValueError):
            Item("A", weight=-1, value=10)

    def test_rejects_negative_value(self) -> None:
        with self.assertRaises(ValueError):
            Item("A", weight=1, value=-1)


if __name__ == "__main__":
    unittest.main()
