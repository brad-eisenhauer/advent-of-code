"""Advent of Code 2022, day 1: https://adventofcode.com/2022/day/1"""
from __future__ import annotations

from io import StringIO
from itertools import islice
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return calc_max_calories(f, 1)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return calc_max_calories(f, 3)


def calc_elf_calories(f: TextIO) -> Iterator[int]:
    elf_calories = 0
    for line in f:
        if line == "\n":
            yield elf_calories
            elf_calories = 0
        else:
            elf_calories += int(line)
    yield elf_calories


def calc_max_calories(f: TextIO, n: int) -> int:
    elf_totals = sorted(calc_elf_calories(f), reverse=True)
    return sum(elf_totals[:n])


SAMPLE_INPUTS = [
    """\
1000
2000
3000

4000

5000
6000

7000
8000
9000

10000
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(("n", "expected"), [(1, 24000), (3, 45000)])
def test_calc_max_calories(sample_input, n, expected):
    assert calc_max_calories(sample_input, n) == expected


def test_calc_elf_calories(sample_input):
    assert list(calc_elf_calories(sample_input)) == [6000, 4000, 11000, 24000, 10000]
