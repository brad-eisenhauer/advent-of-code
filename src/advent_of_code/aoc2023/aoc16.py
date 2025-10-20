"""Advent of Code 2023, day 16: https://adventofcode.com/2023/day/16"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            grid = [line.strip() for line in fp]
        tracer = Tracer(grid)
        return len(tracer.trace())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            grid = [line.strip() for line in fp]
        tracer = Tracer(grid)
        result = 0
        for row in range(len(grid)):
            result = max(result, len(tracer.trace((row, 0), (0, 1))))
            result = max(result, len(tracer.trace((row, len(grid[0]) - 1), (0, -1))))
        for col in range(len(grid[0])):
            result = max(result, len(tracer.trace((0, col), (1, 0))))
            result = max(result, len(tracer.trace((len(grid) - 1, col), (-1, 0))))
        return result


Vector = tuple[int, int]


@dataclass
class Tracer:
    grid: list[str]

    def trace(self, entering: Vector = (0, 0), orientation: Vector = (0, 1)) -> set[Vector]:
        visited: dict[Vector, set[Vector]] = defaultdict(set)
        frontier: deque[tuple[Vector, Vector]] = deque([(entering, orientation)])
        while frontier:
            this_node, this_orient = frontier.popleft()
            if this_orient in visited[this_node]:
                continue
            log.debug("Visiting %s directed %s.", this_node, this_orient)
            visited[this_node].add(this_orient)
            row, col = this_node
            match self.grid[row][col]:
                case ".":
                    next_node = tuple(a + b for a, b in zip(this_node, this_orient))
                    if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                        frontier.append((next_node, this_orient))
                case "|" if this_orient in ((1, 0), (-1, 0)):
                    next_node = tuple(a + b for a, b in zip(this_node, this_orient))
                    if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                        frontier.append((next_node, this_orient))
                case "|":
                    for next_orient in ((1, 0), (-1, 0)):
                        next_node = tuple(a + b for a, b in zip(this_node, next_orient))
                        if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                            frontier.append((next_node, next_orient))
                case "-" if this_orient in ((0, 1), (0, -1)):
                    next_node = tuple(a + b for a, b in zip(this_node, this_orient))
                    if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                        frontier.append((next_node, this_orient))
                case "-":
                    for next_orient in ((0, 1), (0, -1)):
                        next_node = tuple(a + b for a, b in zip(this_node, next_orient))
                        if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                            frontier.append((next_node, next_orient))
                case "\\":
                    # mirror orientation around col == row
                    ox, oy = this_orient
                    next_orient = oy, ox
                    next_node = tuple(a + b for a, b in zip(this_node, next_orient))
                    if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                        frontier.append((next_node, next_orient))
                case "/":
                    # mirror orientation around col == -row
                    ox, oy = this_orient
                    next_orient = -oy, -ox
                    next_node = tuple(a + b for a, b in zip(this_node, next_orient))
                    if all(0 <= n < mx for n, mx in zip(next_node, self.dims)):
                        frontier.append((next_node, next_orient))
        return set(visited.keys())

    @cached_property
    def dims(self) -> Vector:
        return len(self.grid), len(self.grid[0])


SAMPLE_INPUTS = [
    """\
.|...\\....
|.-.\\.....
.....|-...
........|.
..........
.........\\
..../.\\\\..
.-.-/..|..
.|....-|.\\
..//.|....
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
    assert solution.solve_part_one(sample_input) == 46


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 51
