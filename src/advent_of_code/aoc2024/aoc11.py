"""Advent of Code 2024, day 11: https://adventofcode.com/2024/day/11"""
from __future__ import annotations

import math
from functools import cache
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            stones = [int(n) for n in reader.read().split()]
        return sum(count_stones(n, 25) for n in stones)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            stones = [int(n) for n in reader.read().split()]
        return sum(count_stones(n, 75) for n in stones)


@cache
def count_stones(value: int, blinks: int) -> int:
    if blinks == 0:
        return 1
    return sum(count_stones(n, blinks - 1) for n in blink(value))


@cache
def blink(value: int) -> list[int]:
    if value == 0:
        return [1]
    if (digit_count := count_digits(value)) % 2 == 0:
        return [value // 10 ** (digit_count // 2), value % 10 ** (digit_count // 2)]
    return [value * 2024]


def count_digits(n: int) -> int:
    return math.floor(math.log10(n)) + 1


SAMPLE_INPUTS = [
    """\
125 17
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 55312


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 65601038650482
