"""Advent of Code 2020, day 17: https://adventofcode.com/2020/day/17"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from itertools import product
from typing import TextIO

import pytest

from advent_of_code.base import Solution

Vector = tuple[int, ...]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            grid = Grid.read(f)
        result = boot(grid)
        return len(result.active_cubes)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            grid = Grid.read(f, 4)
        result = boot(grid)
        return len(result.active_cubes)


@dataclass(frozen=True)
class Grid:
    active_cubes: frozenset[Vector, ...]

    @classmethod
    def read(cls, f: TextIO, dimensions: int = 3) -> Grid:
        active = frozenset(
            (x, y) + (0,) * (dimensions - 2)
            for y, line in enumerate(f)
            for x, char in enumerate(line.strip())
            if char == "#"
        )
        return cls(active)

    def step(self) -> Grid:
        neighbor_counts = self.count_neighbors()
        next_actives: list[Vector] = []
        for vector in self.active_cubes | set(neighbor_counts):
            match neighbor_counts[vector], (vector in self.active_cubes):
                case 3, _:
                    next_actives.append(vector)
                case 2, True:
                    next_actives.append(vector)
                case _:
                    pass
        return Grid(frozenset(next_actives))

    def count_neighbors(self) -> dict[Vector, int]:
        sample, *_ = self.active_cubes
        dimensions = len(sample)
        neighbor_counts: dict[Vector, int] = defaultdict(int)
        for vector in self.active_cubes:
            for offset in product(*([[-1, 0, 1]] * dimensions)):
                if all(n == 0 for n in offset):
                    continue
                neighbor = tuple(a + b for a, b in zip(vector, offset))
                neighbor_counts[neighbor] += 1
        return neighbor_counts


def boot(initial_grid: Grid, step_count: int = 6) -> Grid:
    grid = initial_grid
    for _ in range(step_count):
        grid = grid.step()
    return grid


SAMPLE_INPUTS = [
    """\
.#.
..#
###
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(("dimensions", "expected"), [(3, 112), (4, 848)])
def test_boot(sample_input, dimensions, expected):
    grid = Grid.read(sample_input, dimensions)
    result = boot(grid)
    assert len(result.active_cubes) == expected
