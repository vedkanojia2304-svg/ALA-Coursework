
import math
import pytest

# Import the Assignment1 class from assign01.py
from assign01 import Assignment1


# Test 1: Check whether the mean is calculated correctly
def test_mean():
    v = Assignment1([1, 5, 6])

    # Mean = (1 + 5 + 6) / 3 = 4
    assert v.mean() == 4.0


# Test 2: Check the mean of a vector containing negative values
def test_mean_negative_values():
    v = Assignment1([-2, 0, 2])

    # Mean = (-2 + 0 + 2) / 3 = 0
    assert v.mean() == 0.0


# Test 3: Check whether the demeaned vector has correct values
def test_demean_values():
    v = Assignment1([1, 5, 6])

    # Mean is 4, so subtract 4 from every element
    # Expected demeaned vector = [-3, 1, 2]
    assert v.demean().elements == [-3.0, 1.0, 2.0]


# Test 4: Check whether the mean of a demeaned vector is zero
def test_demean_mean_is_zero():
    v = Assignment1([1, 5, 6])

    # The mean of a demeaned vector should be approximately zero
    # isclose() allows a small floating-point error
    assert math.isclose(v.demean().mean(), 0.0, abs_tol=1e-9)


# Test 5: Check that demeaning does not modify the original vector
def test_demean_does_not_change_original():
    v = Assignment1([1, 5, 6])

    # Save a copy of the original elements
    original = v.elements.copy()

    # Create the demeaned vector
    v.demean()

    # Check that the original vector remains unchanged
    assert v.elements == original


# Test 6: Check whether standard deviation is calculated correctly
def test_std():
    v = Assignment1([1, 5, 6])

    # Expected population standard deviation = sqrt(14 / 3)
    expected = math.sqrt(14 / 3)

    # Compare the calculated and expected values
    # using a tolerance for floating-point errors
    assert math.isclose(v.std(), expected, rel_tol=1e-9)


# Test 7: Check standard deviation when all values are identical
def test_std_identical_values():
    v = Assignment1([5, 5, 5])

    # All values equal their mean, so standard deviation is zero
    assert v.std() == 0.0


# Test 8: Check that adding a constant does not change standard deviation
def test_std_translation_invariance():
    v1 = Assignment1([1, 5, 6])
    v2 = Assignment1([11, 15, 16])

    # Both vectors have the same spread of values
    # Therefore, their standard deviations should be equal
    assert math.isclose(v1.std(), v2.std())


# Test 9: Check that calculating the mean of an empty vector raises an error
def test_mean_empty_vector():
    v = Assignment1([])

    # Mean is undefined for an empty vector
    with pytest.raises(ValueError):
        v.mean()


# Test 10: Check that calculating standard deviation of an empty vector raises an error
def test_std_empty_vector():
    v = Assignment1([])

    # Standard deviation is undefined for an empty vector
    with pytest.raises(ValueError):
        v.std()
