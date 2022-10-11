"""Advent of Code 2020, day 1: https://adventofcode.com/2020/day/1"""
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(1, 2020)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            nums = read_input(f)
        n, m = find_sums_to(nums, 2020)
        return n * m

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            nums = read_input(f)
        n, m, o = find_sums_to(nums, 2020, 3)
        return n * m * o


def read_input(f: TextIO) -> list[int]:
    return [int(n) for n in f]


def find_sums_to(nums: list[int], m: int, n: int = 2) -> Optional[list[int]]:
    if n == 1:
        return [m] if m in nums else None
    for i, num in enumerate(nums):
        sub = find_sums_to(nums[i + 1 :], m - num, n - 1)
        if sub is not None:
            return [num] + sub
    return None


SAMPLE_INPUT = """\
1721
979
366
299
675
1456
"""


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_read_input(sample_input):
    values = read_input(sample_input)
    assert values == [1721, 979, 366, 299, 675, 1456]


def test_find_sums_to(sample_input):
    values = read_input(sample_input)
    assert find_sums_to(values, 2020) == [1721, 299]
