"""Advent of Code 2021, Day 15: https://adventofcode.com/2021/day/15"""

import contextlib
import logging
from io import StringIO
from typing import Iterator, Optional, Sequence, TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar

Point = tuple[int, int]

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            costs = read_costs(f)
        return Solver(costs).find_min_cost_to_goal((0, 0))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            costs = read_costs(f)
        costs = expand_costs(costs)
        return Solver(costs).find_min_cost_to_goal((0, 0))


EXPANSION_MULTIPLIER = 5


def read_costs(fp: TextIO) -> Sequence[Sequence[int]]:
    return [list(map(int, line.strip())) for line in fp]


def expand_costs(
    costs: Sequence[Sequence[int]], multiplier: int = EXPANSION_MULTIPLIER
) -> Sequence[Sequence[int]]:
    orig_size = len(costs)
    new_size = orig_size * multiplier
    result = [
        [
            (costs[base_row][base_col] - 1 + row // orig_size + col // orig_size) % 9 + 1
            for col in range(new_size)
            for base_row in (row % orig_size,)
            for base_col in (col % orig_size,)
        ]
        for row in range(new_size)
    ]
    return result


class Solver(AStar[Point]):
    def __init__(self, costs: Sequence[Sequence[int]], end: Optional[Point] = None):
        self.costs = costs
        self.end = end or (len(costs) - 1, len(costs[-1]) - 1)

    def is_goal_state(self, point: Point) -> bool:
        return point == self.end

    def heuristic(self, point: Point) -> int:
        return sum(abs(a - b) for a, b in zip(point, self.end))

    def generate_next_states(self, point: Point) -> Iterator[tuple[int, Point]]:
        for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = (a + b for a, b in zip(point, delta))
            if x < 0 or y < 0:
                continue
            with contextlib.suppress(IndexError):
                yield self.costs[x][y], (x, y)


SAMPLE_INPUT = """\
1163751742
1381373672
2136511328
3694931569
7463417111
1319128137
1359912421
3125421639
1293138521
2311944581
"""


@pytest.fixture
def sample_input() -> TextIO:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(("expand_input", "expected"), ((False, 40), (True, 315)))
def test_least_cost_path(sample_input, expand_input, expected):
    costs = read_costs(sample_input)
    if expand_input:
        costs = expand_costs(costs)
    result = Solver(costs).find_min_cost_to_goal((0, 0))
    assert result == expected
