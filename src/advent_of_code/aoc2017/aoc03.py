"""Advent of Code 2017, day 3: https://adventofcode.com/2017/day/3"""

from __future__ import annotations

from collections import defaultdict
from io import StringIO
from itertools import count
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.vector import Vector2


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            n = int(fp.read())
        return calc_distance_to_origin(n)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            n = int(fp.read())
        for m in fill_sums():
            if m > n:
                return m


def calc_square_size(n: int) -> int:
    for r in count(0):
        area = (2 * r + 1) ** 2
        if n <= area:
            return r


def calc_axis_values(r: int) -> tuple[int, int, int, int]:
    inner_square_area = (2 * r - 1) ** 2
    return tuple(inner_square_area + i * r for i in [1, 3, 5, 7])


def calc_distance_from_axis(n: int, r: int) -> int:
    return min(abs(n - x) for x in calc_axis_values(r))


def calc_distance_to_origin(n: int) -> int:
    r = calc_square_size(n)
    offset = calc_distance_from_axis(n, r)
    return r + offset


def winding_order() -> Iterator[Vector2]:
    for r in count(1):
        for y in range(-r + 1, r + 1):
            yield Vector2((r, y))
        for x in range(r - 1, -r - 1, -1):
            yield Vector2((x, r))
        for y in range(r - 1, -r - 1, -1):
            yield Vector2((-r, y))
        for x in range(-r + 1, r + 1):
            yield Vector2((x, -r))


NEIGHBOR_OFFSETS = [
    Vector2((1, 0)),
    Vector2((1, 1)),
    Vector2((0, 1)),
    Vector2((-1, 1)),
    Vector2((-1, 0)),
    Vector2((-1, -1)),
    Vector2((0, -1)),
    Vector2((1, -1)),
]


def neighbors(v: Vector2) -> Iterator[Vector2]:
    for offset in NEIGHBOR_OFFSETS:
        yield v + offset


def fill_sums() -> Iterator[int]:
    grid: dict[Vector2, int] = defaultdict(int)
    grid[Vector2((0, 0))] = 1
    yield 1

    for v in winding_order():
        neighbor_sum = sum(grid[nv] for nv in neighbors(v))
        grid[v] = neighbor_sum
        log.debug("Setting grid value %d at %s.", neighbor_sum, v)
        yield neighbor_sum


SAMPLE_INPUTS = [
    """\
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == ...


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...
