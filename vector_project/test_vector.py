
# Tests for the Vec class
# Covers vector operations and statistical methods

import unittest
import math
from vec import Vec


class TestVec(unittest.TestCase):

    # -------------------------
    # Vector creation
    # -------------------------

    def test_creation(self):
        v = Vec([1, 2, 3])
        self.assertEqual(v.elements, [1, 2, 3])

    def test_length(self):
        v = Vec([10, 20, 30, 40])
        self.assertEqual(len(v), 4)

    # -------------------------
    # Vector operations
    # -------------------------

    def test_addition(self):
        v1 = Vec([1, 2, 3])
        v2 = Vec([4, 5, 6])
        self.assertEqual((v1 + v2).elements, [5, 7, 9])

    def test_subtraction(self):
        v1 = Vec([1, 2, 3])
        v2 = Vec([4, 5, 6])
        self.assertEqual((v2 - v1).elements, [3, 3, 3])

    def test_scalar_multiplication(self):
        v = Vec([1, 2, 3])
        self.assertEqual((2 * v).elements, [2, 4, 6])

    def test_inplace_multiplication(self):
        v = Vec([1, 2, 3])
        v *= 3
        self.assertEqual(v.elements, [3, 6, 9])

    def test_negation(self):
        v = Vec([1, 2, 3])
        self.assertEqual((-v).elements, [-1, -2, -3])

    def test_inplace_addition(self):
        v = Vec([1, 2, 3])
        v += Vec([4, 5, 6])
        self.assertEqual(v.elements, [5, 7, 9])

    # -------------------------
    # Vector creation methods
    # -------------------------

    def test_zeros(self):
        self.assertEqual(Vec.zeros(5).elements, [0] * 5)

    def test_ones(self):
        self.assertEqual(Vec.ones(4).elements, [1] * 4)

    # -------------------------
    # Norm
    # -------------------------

    def test_norm(self):
        v = Vec([3, 4])
        self.assertAlmostEqual(v.norm(), 5)

    # -------------------------
    # Mean tests
    # -------------------------

    def test_mean(self):
        v = Vec([2, 4, 6, 8])
        self.assertAlmostEqual(v.mean(), 5)

    def test_mean_negative_numbers(self):
        v = Vec([-2, -4, -6])
        self.assertAlmostEqual(v.mean(), -4)

    def test_mean_single_element(self):
        v = Vec([7])
        self.assertAlmostEqual(v.mean(), 7)

    def test_mean_constant_vector(self):
        v = Vec([5, 5, 5])
        self.assertAlmostEqual(v.mean(), 5)

    def test_mean_empty_vector(self):
        v = Vec([])
        with self.assertRaises(ValueError):
            v.mean()

    def test_mean_translation(self):
        v1 = Vec([1, 2, 3])
        v2 = Vec([11, 12, 13])
        self.assertAlmostEqual(v2.mean(), v1.mean() + 10)

    def test_mean_permutation(self):
        v1 = Vec([1, 2, 3, 4])
        v2 = Vec([4, 2, 1, 3])
        self.assertAlmostEqual(v1.mean(), v2.mean())

    # -------------------------
    # Demean tests
    # -------------------------

    def test_demean(self):
        v = Vec([2, 4, 6, 8])
        result = v.demean()
        expected = [-3, -1, 1, 3]

        for actual, wanted in zip(result.elements, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_demean_mean_is_zero(self):
        v = Vec([2, 4, 6, 8])
        self.assertAlmostEqual(v.demean().mean(), 0)

    def test_demean_single_element(self):
        v = Vec([10])
        self.assertEqual(v.demean().elements, [0])

    def test_demean_constant_vector(self):
        v = Vec([5, 5, 5])
        self.assertEqual(v.demean().elements, [0, 0, 0])

    def test_demean_does_not_modify_original(self):
        v = Vec([2, 4, 6, 8])
        original = v.elements.copy()

        result = v.demean()

        self.assertEqual(v.elements, original)
        self.assertIsNot(v, result)

    def test_demean_empty_vector(self):
        v = Vec([])
        with self.assertRaises(ValueError):
            v.demean()

    # -------------------------
    # Standard deviation tests
    # -------------------------

    def test_std(self):
        v = Vec([2, 4, 6, 8])
        self.assertAlmostEqual(v.std(), math.sqrt(5))

    def test_std_single_element(self):
        v = Vec([10])
        self.assertAlmostEqual(v.std(), 0)

    def test_std_constant_vector(self):
        v = Vec([5, 5, 5, 5])
        self.assertAlmostEqual(v.std(), 0)

    def test_std_non_negative(self):
        v = Vec([-2, 4, 6, 8])
        self.assertGreaterEqual(v.std(), 0)

    def test_std_empty_vector(self):
        v = Vec([])
        with self.assertRaises(ValueError):
            v.std()

    def test_std_translation_invariance(self):
        v1 = Vec([1, 2, 3, 4])
        v2 = Vec([11, 12, 13, 14])
        self.assertAlmostEqual(v1.std(), v2.std())

    def test_std_permutation_invariance(self):
        v1 = Vec([1, 2, 3, 4])
        v2 = Vec([4, 2, 1, 3])
        self.assertAlmostEqual(v1.std(), v2.std())

    def test_std_scaling(self):
        v = Vec([1, 2, 3, 4])
        scaled = 3 * v
        self.assertAlmostEqual(scaled.std(), 3 * v.std())

    def test_std_negative_scaling(self):
        v = Vec([1, 2, 3, 4])
        scaled = -2 * v
        self.assertAlmostEqual(scaled.std(), 2 * v.std())

    def test_std_does_not_modify_original(self):
        v = Vec([1, 2, 3, 4])
        original = v.elements.copy()

        v.std()

        self.assertEqual(v.elements, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)