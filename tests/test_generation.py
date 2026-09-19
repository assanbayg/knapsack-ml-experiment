import unittest

from knapsack_ml_experiment import (
    generate_independent_instance,
    generate_independent_instances,
    generate_similar_ratio_instance,
    generate_similar_ratio_instances,
    generate_strongly_correlated_instance,
    generate_strongly_correlated_instances,
    generate_weakly_correlated_instance,
    generate_weakly_correlated_instances,
)


class TestIndependentInstanceGeneration(unittest.TestCase):
    def test_generates_requested_number_of_items_within_ranges(self) -> None:
        items, capacity = generate_independent_instance(
            number_of_items=5,
            seed=12,
            weight_range=(2, 8),
            value_range=(10, 20),
            capacity_ratio=0.4,
        )

        self.assertEqual(len(items), 5)
        self.assertEqual(
            [item.name for item in items],
            ["item_0", "item_1", "item_2", "item_3", "item_4"],
        )
        self.assertTrue(all(2 <= item.weight <= 8 for item in items))
        self.assertTrue(all(10 <= item.value <= 20 for item in items))
        self.assertEqual(capacity, round(sum(item.weight for item in items) * 0.4))

    def test_same_seed_reproduces_same_instance(self) -> None:
        first = generate_independent_instance(number_of_items=10, seed=7)
        second = generate_independent_instance(number_of_items=10, seed=7)

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_instances(self) -> None:
        first = generate_independent_instance(number_of_items=10, seed=7)
        second = generate_independent_instance(number_of_items=10, seed=8)

        self.assertNotEqual(first, second)

    def test_rejects_invalid_parameters(self) -> None:
        invalid_arguments = [
            {"number_of_items": 0, "seed": 1},
            {"number_of_items": 1, "seed": 1, "weight_range": (0, 5)},
            {"number_of_items": 1, "seed": 1, "weight_range": (5, 4)},
            {"number_of_items": 1, "seed": 1, "value_range": (0, 5)},
            {"number_of_items": 1, "seed": 1, "value_range": (5, 4)},
            {"number_of_items": 1, "seed": 1, "capacity_ratio": 0},
            {"number_of_items": 1, "seed": 1, "capacity_ratio": 1},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    generate_independent_instance(**arguments)


class TestIndependentBatchGeneration(unittest.TestCase):
    def test_generates_one_thousand_instances_by_default(self) -> None:
        instances = generate_independent_instances()

        self.assertEqual(len(instances), 1_000)
        self.assertEqual(instances[0].seed, 0)
        self.assertEqual(instances[-1].seed, 999)
        self.assertTrue(all(len(instance.items) == 20 for instance in instances))
        self.assertTrue(
            all(instance.family == "independent" for instance in instances)
        )

    def test_batch_is_reproducible_and_uses_consecutive_seeds(self) -> None:
        first = generate_independent_instances(
            3, number_of_items=4, starting_seed=100
        )
        second = generate_independent_instances(
            3, number_of_items=4, starting_seed=100
        )

        self.assertEqual(first, second)
        self.assertEqual([instance.seed for instance in first], [100, 101, 102])

    def test_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            generate_independent_instances(0)


class TestOtherInstanceFamilies(unittest.TestCase):
    def test_strongly_correlated_values_have_fixed_offset(self) -> None:
        items, capacity = generate_strongly_correlated_instance(
            20, seed=4, value_offset=7
        )

        self.assertTrue(all(item.value == item.weight + 7 for item in items))
        self.assertEqual(capacity, round(sum(item.weight for item in items) * 0.5))

    def test_weakly_correlated_values_stay_within_spread(self) -> None:
        items, _ = generate_weakly_correlated_instance(
            20, seed=4, value_spread=6
        )

        self.assertTrue(
            all(abs(item.value - item.weight) <= 6 for item in items)
        )
        self.assertTrue(all(item.value > 0 for item in items))

    def test_similar_ratio_values_stay_near_target(self) -> None:
        items, _ = generate_similar_ratio_instance(
            20,
            seed=4,
            weight_range=(20, 50),
            target_ratio=3.0,
            ratio_jitter=0.05,
        )

        ratios = [item.value / item.weight for item in items]
        self.assertTrue(all(2.8 <= ratio <= 3.2 for ratio in ratios))

    def test_each_generator_is_reproducible(self) -> None:
        generators = (
            generate_strongly_correlated_instance,
            generate_weakly_correlated_instance,
            generate_similar_ratio_instance,
        )

        for generator in generators:
            with self.subTest(generator=generator.__name__):
                self.assertEqual(generator(10, 9), generator(10, 9))

    def test_batches_record_seed_and_family(self) -> None:
        cases = (
            (generate_strongly_correlated_instances, "strongly_correlated"),
            (generate_weakly_correlated_instances, "weakly_correlated"),
            (generate_similar_ratio_instances, "similar_ratio"),
        )

        for generator, expected_family in cases:
            with self.subTest(family=expected_family):
                instances = generator(3, number_of_items=5, starting_seed=20)
                self.assertEqual([instance.seed for instance in instances], [20, 21, 22])
                self.assertTrue(
                    all(instance.family == expected_family for instance in instances)
                )

    def test_rejects_family_specific_invalid_parameters(self) -> None:
        invalid_calls = (
            lambda: generate_strongly_correlated_instance(5, 1, value_offset=-1),
            lambda: generate_weakly_correlated_instance(5, 1, value_spread=-1),
            lambda: generate_similar_ratio_instance(5, 1, target_ratio=0),
            lambda: generate_similar_ratio_instance(5, 1, ratio_jitter=1),
            lambda: generate_similar_ratio_instances(0),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
