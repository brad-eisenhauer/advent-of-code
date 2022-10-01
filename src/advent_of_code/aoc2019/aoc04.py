""" Advent of Code 2019, Day 04: https://adventofcode.com/2019/day/4 """
from functools import cache
from io import StringIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(4, 2019)
        self.read_valid_range = cache(self._read_valid_range)

    def _read_valid_range(self):
        with self.open_input() as f:
            return get_valid_range(f.readline().strip())

    def solve_part_one(self) -> int:
        return sum(1 for n in self.read_valid_range() if code_is_valid(n))

    def solve_part_two(self) -> int:
        return sum(1 for n in self.read_valid_range() if code_is_valid(n, strict=True))


def get_valid_range(s: str) -> range:
    left, right = map(int, s.split("-"))
    return range(left, right + 1)


def code_is_valid(n: int, strict: bool = False) -> bool:
    # six digit number; no leading zeroes
    if n not in range(100000, 1000000):
        return False

    def generate_digits():
        temp = n
        while temp > 0:
            yield temp % 10
            temp //= 10

    # digits must be monotonically increasing
    digits = generate_digits()
    last_digit = next(digits)
    for next_digit in digits:
        if next_digit > last_digit:
            return False
        last_digit = next_digit

    # number must contain at least two consecutive digits (exactly two if strict)
    digits = generate_digits()
    last_digit = next(digits)
    consec_count = 1
    for next_digit in digits:
        if next_digit == last_digit:
            consec_count += 1
            if not strict and consec_count > 1:
                break
        else:
            if consec_count == 2:
                break
            consec_count = 1
        last_digit = next_digit
    else:
        if consec_count != 2:
            return False

    return True


SAMPLE_INPUT = """\
271973-785961
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(
    "n,strict,is_valid",
    (
        (111111, False, True),
        (111111, True, False),
        (223450, True, False),
        (123789, True, False),
        (112233, True, True),
        (123444, True, False),
        (111122, True, True),
    ),
)
def test_code_is_valid(n: int, strict: bool, is_valid: bool):
    assert code_is_valid(n, strict) == is_valid


def test_get_valid_range(sample_input):
    assert get_valid_range(sample_input.readline().strip()) == range(271973, 785962)
