"""Advent of Code 2022, day 18: https://adventofcode.com/2022/day/18"""

from __future__ import annotations

from collections import deque
from io import StringIO
from typing import Iterable, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(18, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            locs = parse_droplet(f)
        return calc_surface_area(locs)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            locs = parse_droplet(f)
        return calc_exterior_surface_area(locs)


Vector = tuple[int, ...]
ADJACENT_DIRS = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]


def calc_surface_area(locs: set[Vector]) -> int:
    result = 0
    for loc in locs:
        for d in ADJACENT_DIRS:
            adj = tuple(a + b for a, b in zip(loc, d))
            if adj not in locs:
                result += 1
    return result


def calc_exterior_surface_area(locs: set[Vector]) -> int:
    exterior_locs = find_exterior_locs(locs)
    result = 0
    for loc in locs:
        for d in ADJACENT_DIRS:
            adj = tuple(a + b for a, b in zip(loc, d))
            if adj in exterior_locs:
                result += 1
    return result


def find_exterior_locs(locs: set[Vector]) -> Iterable[Vector]:
    max_bounds = tuple(max(ns) + 1 for ns in zip(*locs))
    min_bounds = tuple(min(ns) - 1 for ns in zip(*locs))
    exterior_locs: set[Vector] = set()

    frontier = deque([min_bounds])
    while frontier:
        loc = frontier.popleft()
        if loc in exterior_locs:
            continue
        exterior_locs.add(loc)
        for d in ADJACENT_DIRS:
            adj_loc = tuple(a + b for a, b in zip(loc, d))
            if adj_loc in locs:
                continue
            if all(mn <= my_loc <= mx for mn, my_loc, mx in zip(min_bounds, adj_loc, max_bounds)):
                frontier.append(adj_loc)

    return exterior_locs


def parse_droplet(f: TextIO) -> set[Vector]:
    locs: set[Vector] = set()
    for line in f:
        locs.add(tuple(int(n) for n in line.split(",")))
    return locs


SAMPLE_INPUTS = [
    """\
2,2,2
1,2,2
3,2,2
2,1,2
2,3,2
2,2,1
2,2,3
2,2,4
2,2,6
1,2,5
3,2,5
2,1,5
2,3,5
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_calc_surface_area(sample_input):
    locs = parse_droplet(sample_input)
    assert calc_surface_area(locs) == 64


def test_calc_exterior_surface_area(sample_input):
    locs = parse_droplet(sample_input)
    assert calc_exterior_surface_area(locs) == 58
