""" Advent of Code 2021, Day 09: https://adventofcode.com/2021/day/9 """
from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, reduce
from io import StringIO
from itertools import islice, product
from typing import Iterable, Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution

Point = tuple[int, int]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            height_map = HeightMap.read(f)
        return height_map.calc_total_risk_score()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            height_map = HeightMap.read(f)
        return reduce(operator.mul, height_map.calc_largest_basin_sizes(3))


@dataclass
class Basin:
    members: set[Point]
    low: Point


class Direction(tuple, Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @cached_property
    def opposite(self):
        return {
            self.UP: self.DOWN,
            self.DOWN: self.UP,
            self.LEFT: self.RIGHT,
            self.RIGHT: self.LEFT,
        }[self]

    def from_point(self, point: Point) -> Point:
        row, col = point
        row_offset, col_offset = self.value
        return row + row_offset, col + col_offset


class HeightMap:
    def __init__(self, heights: Iterable[Iterable[int]]):
        # pad heights with border of 9s
        heights = tuple((9, *row, 9) for row in heights)
        map_width = len(heights[0])
        self.heights = ((9,) * map_width, *heights, (9,) * map_width)

    @classmethod
    def read(cls, fp: TextIO) -> HeightMap:
        return cls((int(n) for n in line.strip()) for line in fp)

    def get_height(self, point: Point) -> int:
        row, col = point
        return self.heights[row][col]

    @cached_property
    def basins(self) -> list[Basin]:
        result = []

        row_range = range(1, len(self.heights) - 1)
        col_range = range(1, len(self.heights[0]) - 1)

        all_basin_points = set()

        for point in product(row_range, col_range):
            if point in all_basin_points or self.get_height(point) == 9:
                continue
            if (low_point := self.find_low_point(point)) is not None:
                basin = Basin(members=set(self.find_basin_points(low_point)), low=low_point)
                result.append(basin)
                all_basin_points |= basin.members

        return result

    def find_low_point(self, start: Point) -> Optional[Point]:
        value = self.get_height(start)
        if value == 9:
            return None

        try:
            downstream_point = next(
                p for d in Direction for p in (d.from_point(start),) if self.get_height(p) < value
            )
            return self.find_low_point(downstream_point)
        except StopIteration:
            return start

    def find_basin_points(self, start: Point) -> Iterator[Point]:
        """BFS for all points until we hit the wall of 9s"""
        queue = [start]
        visited = set()

        while len(queue) > 0:
            if (point := queue.pop(0)) in visited:
                continue
            visited.add(point)
            if self.get_height(point) < 9:
                yield point
                queue.extend(d.from_point(point) for d in Direction)

    def calc_total_risk_score(self) -> int:
        return sum(self.get_height(basin.low) + 1 for basin in self.basins)

    def calc_largest_basin_sizes(self, n: int) -> Iterator[int]:
        basin_sizes = (len(basin.members) for basin in self.basins)
        return islice(sorted(basin_sizes, reverse=True), n)


TEST_INPUT = """2199943210
3987894921
9856789892
8767896789
9899965678
"""


@pytest.fixture()
def height_map():
    with StringIO(TEST_INPUT) as fp:
        height_map = HeightMap.read(fp)
    return height_map


def test_calc_total_risk_score(height_map):
    result = height_map.calc_total_risk_score()
    assert result == 15


def test_sum_3_largest_basins(height_map):
    result = reduce(operator.mul, height_map.calc_largest_basin_sizes(3))
    assert result == 1134
