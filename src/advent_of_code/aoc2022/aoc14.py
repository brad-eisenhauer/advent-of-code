"""Advent of Code 2022, day 14: https://adventofcode.com/2022/day/14"""
from __future__ import annotations

from functools import cache
from io import StringIO
from typing import Iterable, Optional, TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import clamp


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            cave = Cave.parse(f)
        cave.run_sim()
        return len(cave.sand)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            cave = Cave.parse(f, with_floor=True)
        cave.run_sim()
        return len(cave.sand)


Vector = tuple[int, ...]


class Cave:
    FALLS: Iterable[Vector] = ((0, 1), (-1, 1), (1, 1))
    START: Vector = (500, 0)

    def __init__(self, walls: set[Vector], has_floor: bool = False):
        self.walls = walls
        self.sand = set()
        self.floor = 2 + max(y for _, y in walls) if has_floor else None
        self.came_from: dict[Vector, Vector] = {}

        self.max_wall_y = cache(self._max_wall_y)

    def __eq__(self, other):
        return isinstance(other, Cave) and self.walls == other.walls

    @classmethod
    def parse(cls, f: TextIO, with_floor: bool = False) -> Cave:
        walls: set[Vector] = set()

        def add_walls_between(left: Vector, right: Vector):
            unit_delta: Vector = tuple(clamp(r - l, -1, 1) for l, r in zip(left, right))
            v = left
            while v != right:
                walls.add(v)
                v = tuple(a + b for a, b in zip(v, unit_delta))
            walls.add(right)

        for line in f:
            vertices = line.rstrip().split(" -> ")
            for left, right in zip(vertices, vertices[1:]):
                left = tuple(int(n) for n in left.split(","))
                right = tuple(int(n) for n in right.split(","))
                add_walls_between(left, right)

        return Cave(walls, with_floor)

    def _max_wall_y(self, at_x: int) -> int:
        if self.floor is not None:
            return self.floor
        try:
            return max(y for x, y in self.walls if x == at_x)
        except ValueError:
            return -1

    def is_obstacle(self, pos: Vector) -> bool:
        if self.floor is not None and pos[1] == self.floor:
            return True
        return pos in self.walls or pos in self.sand

    def drop_sand(self, start: Vector) -> Optional[Vector]:
        loc = start
        while loc[1] < self.max_wall_y(loc[0]):
            for fall in self.FALLS:
                next_loc = tuple(a + b for a, b in zip(loc, fall))
                if not self.is_obstacle(next_loc):
                    self.came_from[next_loc] = loc
                    loc = next_loc
                    break
            else:
                return loc

        return None

    def run_sim(self):
        start_loc = self.START
        while (resting_loc := self.drop_sand(start_loc)) is not None:
            self.sand.add(resting_loc)
            if resting_loc in self.came_from:
                start_loc = self.came_from[resting_loc]
            else:
                break


SAMPLE_INPUTS = [
    """\
498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_cave_parse(sample_input):
    expected = Cave(
        {
            *((498, y) for y in range(4, 7)),
            *((x, 6) for x in range(496, 499)),
            *((x, 4) for x in range(502, 504)),
            *((502, y) for y in range(4, 10)),
            *((x, 9) for x in range(494, 503)),
        }
    )
    assert Cave.parse(sample_input) == expected


@pytest.mark.parametrize(("with_floor", "expected"), [(False, 24), (True, 93)])
def test_run_sim(sample_input, with_floor, expected):
    cave = Cave.parse(sample_input, with_floor=with_floor)
    cave.run_sim()
    assert len(cave.sand) == expected
