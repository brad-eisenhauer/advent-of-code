""" Advent of Code 2019, Day 01: https://adventofcode.com/2019/day/1 """
from functools import cache
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2019, **kwargs)
        self.read_masses = cache(self._read_masses)

    def _read_masses(self) -> list[int]:
        with self.open_input() as f:
            return [int(m) for m in f.readlines()]

    def solve_part_one(self) -> int:
        return sum(calc_fuel_requirement(m) for m in self.read_masses())

    def solve_part_two(self) -> int:
        return sum(calc_total_fuel(m) for m in self.read_masses())


def parse_input(fp: TextIO) -> Iterator[int]:
    for line in fp:
        yield int(line.strip())


def calc_fuel_requirement(mass: int) -> int:
    return max(0, mass // 3 - 2)


def calc_total_fuel(mass: int) -> int:
    added_fuel = calc_fuel_requirement(mass)
    result = added_fuel
    while (added_fuel := calc_fuel_requirement(added_fuel)) > 0:
        result += added_fuel
    return result


SAMPLE_INPUT = """\
12
14
1969
100756
"""


@pytest.fixture()
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(("mass", "expected"), ((12, 2), (14, 2), (1969, 654), (100756, 33583)))
def test_calc_fuel_requirement(mass, expected):
    assert calc_fuel_requirement(mass) == expected


@pytest.mark.parametrize(("mass", "expected"), ((12, 2), (14, 2), (1969, 966), (100756, 50346)))
def test_calc_total_fuel(mass, expected):
    assert calc_total_fuel(mass) == expected
