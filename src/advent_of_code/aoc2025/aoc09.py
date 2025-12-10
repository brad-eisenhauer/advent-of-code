"""Advent of Code 2025, day 9: https://adventofcode.com/2025/day/9"""

from __future__ import annotations

from functools import reduce
from itertools import combinations
import logging
from io import StringIO
import operator
from typing import IO, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            corners = read_corners(fp)
        return find_largest_rectangle(corners)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ...


Corner: TypeAlias = tuple[int, int]


def read_corners(reader: IO) -> list[Corner]:
    return list(eval(line) for line in reader)


def find_largest_rectangle(corners: list[Corner]) -> int:
    result = 0
    for left, right in combinations(corners, 2):
        dims = [abs(a - b) + 1 for a, b in zip(left, right)]
        result = max(result, reduce(operator.mul, dims))
    return result


SAMPLE_INPUTS = [
    """\
7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 50


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 24
