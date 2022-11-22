"""Advent of Code 2015, day 2: https://adventofcode.com/2015/day/2"""
from __future__ import annotations

import operator
from functools import reduce
from io import StringIO
from itertools import combinations, product
from typing import Callable, Collection

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2015, **kwargs)

    def solve_part_one(self) -> int:
        return self._solve(calc_required_area)

    def solve_part_two(self) -> int:
        return self._solve(calc_required_ribbon)

    def _solve(self, fn: Callable[[Collection[int]], int]) -> int:
        with self.open_input() as f:
            result = 0
            for line in f:
                dims = parse_dims(line)
                result += fn(dims)
        return result


def calc_required_area(dims: Collection[int]) -> int:
    side_areas = [a * b for a, b in combinations(dims, 2)]
    return 2 * sum(side_areas) + min(side_areas)


def parse_dims(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in text.split("x"))


def calc_required_ribbon(dims: Collection[int]) -> int:
    perimeters = [2 * (a + b) for a, b in combinations(dims, 2)]
    return min(perimeters) + reduce(operator.mul, dims)


SAMPLE_INPUTS = [
    """\
2x3x4
""",
    """\
1x1x10
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, (2, 3, 4)), (1, (1, 1, 10))], indirect=["sample_input"]
)
def test_parse_dims(sample_input, expected):
    assert parse_dims(sample_input.readline()) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 58), (1, 43)], indirect=["sample_input"]
)
def test_required_area(sample_input, expected):
    dims = parse_dims(sample_input.readline())
    assert calc_required_area(dims) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 34), (1, 14)], indirect=["sample_input"]
)
def test_required_ribbon(sample_input, expected):
    dims = parse_dims(sample_input.readline())
    assert calc_required_ribbon(dims) == expected
