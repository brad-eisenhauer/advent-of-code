# test/util/test_math.py
import pytest

from advent_of_code.util.math import div


@pytest.mark.parametrize(
    "numerator, denominator, expected_quotient, expected_remainder",
    [
        # Positive numbers
        (10, 3, 3, 1),
        (15, 4, 3, 3),
        # Negative numbers
        (-10, 3, -3, -1),  # -10 / 3 = -3.333... → quotient = -3, remainder = -1
        (10, -3, -3, 1),  # 10 / -3 = -3.333... → quotient = -3, remainder = 1
        (-10, -3, 3, -1),  # -10 / -3 = 3.333... → quotient = 3, remainder = -1
        # Zero numerator
        (0, 5, 0, 0),
        (0, -2, 0, 0),
        # Large numbers
        (999999999999, 100000000000, 9, 99999999999),
        # Equal numerator and denominator
        (7, 7, 1, 0),
        # Remainder is zero
        (12, 4, 3, 0),
        # Remainder is non-zero
        (11, 4, 2, 3),
    ],
)
def test_div(numerator, denominator, expected_quotient, expected_remainder):
    """Test the div function with various inputs."""
    quotient, remainder = div(numerator, denominator)
    assert quotient == expected_quotient
    assert remainder == expected_remainder


def test_div_zero_denominator():
    """Test division by zero (if the function raises an exception)."""
    with pytest.raises(ZeroDivisionError):
        div(10, 0)
