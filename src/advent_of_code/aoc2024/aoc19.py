"""Advent of Code 2024, day 19: https://adventofcode.com/2024/day/19"""

from __future__ import annotations

from functools import cache
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(19, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as reader:
            available_towels = tuple(t.strip() for t in reader.readline().split(", "))
            reader.readline()
            for design in reader:
                if is_design_possible(design.strip(), available_towels):
                    result += 1
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as reader:
            available_towels = tuple(t.strip() for t in reader.readline().split(", "))
            reader.readline()
            for design in reader:
                result += count_arrangements(design.strip(), available_towels)
        return result


@cache
def is_design_possible(design: str, available_towels: tuple[str]) -> bool:
    if design == "":
        return True
    for towel in available_towels:
        if design.startswith(towel) and is_design_possible(design[len(towel) :], available_towels):
            return True
    return False


@cache
def count_arrangements(design: str, available_towels: tuple[str]) -> int:
    if design == "":
        return 1
    result = 0
    for towel in available_towels:
        if design.startswith(towel):
            result += count_arrangements(design[len(towel) :], available_towels)
    return result


SAMPLE_INPUTS = [
    """\
r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 6


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 16
