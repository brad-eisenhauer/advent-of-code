"""Advent of Code 2015, day 20: https://adventofcode.com/2015/day/20"""
from __future__ import annotations

from itertools import count

import pytest

from advent_of_code.base import Solution
from advent_of_code.util import math as aum


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(20, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            min_present_count = int(f.read().strip())
        return next(
            house_number
            for house_number in count(min_present_count // 50)
            if calc_presents_p1(house_number) >= min_present_count
        )

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            min_present_count = int(f.read().strip())
        return next(
            house_number
            for house_number in count(min_present_count // 55)
            if calc_presents_p2(house_number) >= min_present_count
        )


def calc_presents_p1(house_number: int) -> int:
    return 10 * aum.sum_of_factors(house_number)


def calc_presents_p2(house_number: int) -> int:
    sum_of_factors = sum(f for f in aum.all_factors(house_number) if house_number // f <= 50)
    return 11 * sum_of_factors


@pytest.mark.parametrize(
    ("house_number", "expected"),
    [(1, 10), (2, 30), (3, 40), (4, 70), (5, 60), (6, 120), (7, 80), (8, 150), (9, 130)],
)
def test_calc_presents(house_number, expected):
    assert calc_presents_p1(house_number) == expected
