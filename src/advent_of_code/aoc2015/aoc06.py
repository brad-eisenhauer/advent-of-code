"""Advent of Code 2015, day 6: https://adventofcode.com/2015/day/6"""
from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return count_on_at_end(f)


def count_on_at_end(instructions: TextIO) -> int:
    pattern = re.compile(r"^(turn on|turn off|toggle) (\d+),(\d+) through (\d+),(\d+)")
    grid = NULL_GRID
    for line in instructions:
        op, x1, y1, x2, y2 = pattern.match(line.strip()).groups()
        other = BasicGrid(range(int(x1), int(x2) + 1), range(int(y1), int(y2) + 1))
        match op:
            case "turn on":
                grid = grid.turn_on(other)
            case "turn off":
                grid = grid.turn_off(other)
            case "toggle":
                grid = grid.toggle(other)
    return grid.count()


class Grid:
    @abstractmethod
    def count(self) -> int:
        ...

    def turn_on(self, grid: BasicGrid) -> Grid:
        return self | grid

    def turn_off(self, grid: BasicGrid) -> Grid:
        return self - grid

    def toggle(self, grid: BasicGrid) -> Grid:
        return self ^ grid

    @abstractmethod
    def __and__(self, other):
        ...

    def __or__(self, other):
        if other is NULL_GRID:
            return self
        return UnionGrid(self, other)

    def __xor__(self, other):
        if other is NULL_GRID:
            return self
        return XorGrid(self, other)

    def __sub__(self, other):
        if other is NULL_GRID:
            return self
        return DifferenceGrid(self, other)


class _NullGrid(Grid):
    _instance: Optional[_NullGrid] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def count(self) -> int:
        return 0

    def __and__(self, other) -> Grid:
        return self._instance

    def __or__(self, other) -> Grid:
        return other

    def __xor__(self, other):
        return other

    def __sub__(self, other):
        return self._instance


NULL_GRID = _NullGrid()


@dataclass(frozen=True)
class BasicGrid(Grid):
    xs: range
    ys: range

    def count(self) -> int:
        return len(self.xs) * len(self.ys)

    def __and__(self, g: Grid) -> Grid:
        if isinstance(g, BasicGrid):
            x_min = max([min(self.xs), min(g.xs)])
            x_max = min([max(self.xs), max(g.xs)])
            y_min = max([min(self.ys), min(g.ys)])
            y_max = min([max(self.ys), max(g.ys)])
            if x_max < x_min or y_max < y_min:
                return NULL_GRID
            return BasicGrid(range(x_min, x_max + 1), range(y_min, y_max + 1))
        return g & self


@dataclass(frozen=True)
class UnionGrid(Grid):
    left: Grid
    right: Grid

    def count(self) -> int:
        return self.left.count() + self.right.count() - (self.left & self.right).count()

    def __and__(self, g: Grid) -> Grid:
        left_int = self.left & g
        right_int = self.right & g
        if left_int is NULL_GRID:
            return right_int
        if right_int is NULL_GRID:
            return left_int
        return left_int | right_int


@dataclass(frozen=True)
class DifferenceGrid(Grid):
    minuend: Grid
    subtrahend: Grid

    def count(self) -> int:
        return self.minuend.count() - (self.minuend & self.subtrahend).count()

    def __and__(self, g: Grid) -> Grid:
        return (self.minuend & g) - self.subtrahend


@dataclass(frozen=True)
class XorGrid(Grid):
    left: Grid
    right: Grid

    def count(self) -> int:
        return (
            self.left.count() + self.right.count() - 2 * (self.left & self.right).count()
        )

    def __and__(self, g: Grid) -> Grid:
        left_part = self.left - self.right
        right_part = self.right - self.left
        if left_part is NULL_GRID:
            return right_part & g
        if right_part is NULL_GRID:
            return left_part & g
        return (left_part | right_part) & g


SAMPLE_INPUTS = [
    """\
turn on 0,0 through 999,999
toggle 0,0 through 999,0
turn off 499,499 through 500,500
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_count_on_at_end(sample_input):
    assert count_on_at_end(sample_input) == 998996
