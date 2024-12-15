"""Advent of Code 2017, day 11: https://adventofcode.com/2017/day/11"""

from __future__ import annotations

from io import StringIO
from typing import IO, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            directions = reader.read().strip()
        loc = (0, 0, 0)
        for d in directions.split(","):
            loc = vector_add(loc, GRID_DIRECTIONS[d])
        return calc_min_steps(loc)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            directions = reader.read().strip()
        loc = (0, 0, 0)
        result = 0
        for d in directions.split(","):
            loc = vector_add(loc, GRID_DIRECTIONS[d])
            result = max(result, calc_min_steps(loc))
        return result


Vector: TypeAlias = tuple[int, int, int]  # Using cubic coordinates for hex grid


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(l + r for l, r in zip(left, right))


GRID_DIRECTIONS = {
    "n": (0, -1, 1),
    "ne": (1, -1, 0),
    "se": (1, 0, -1),
    "s": (0, 1, -1),
    "sw": (-1, 1, 0),
    "nw": (-1, 0, 1),
}


def calc_min_steps(loc: Vector) -> int:
    # calc min manhattan distance along any two axes
    return sum(sorted(abs(n) for n in loc)[:2])


SAMPLE_INPUTS = [
    """\
ne,ne,ne
""",
    """\
ne,ne,sw,sw
""",
    """\
ne,ne,s,s
""",
    """\
se,sw,se,sw,sw
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 3), (1, 0), (2, 2), (3, 3)],
    indirect=["sample_input"],
)
def test_part_one(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 3), (1, 2), (2, 2), (3, 3)],
    indirect=["sample_input"],
)
def test_part_two(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_two(sample_input) == expected
