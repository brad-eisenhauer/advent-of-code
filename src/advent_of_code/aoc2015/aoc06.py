"""Advent of Code 2015, day 6: https://adventofcode.com/2015/day/6"""
from __future__ import annotations

import re

# import sys
from abc import abstractmethod
from dataclasses import dataclass
from io import StringIO
from itertools import combinations
from typing import ClassVar, Optional, TextIO, TypeVar

import pytest

from advent_of_code.base import Solution

# sys.setrecursionlimit(5000)
T = TypeVar("T")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return count_on_at_end(f)


def count_on_at_end(instructions: TextIO) -> int:
    pattern = re.compile(r"^(turn on|turn off|toggle) (\d+),(\d+) through (\d+),(\d+)")
    grid = BruteForceGrid()
    for line in instructions:
        op, *coords = pattern.match(line.strip()).groups()
        x1, y1, x2, y2 = (int(n) for n in coords)
        other = BasicGrid(range(x1, x2 + 1), range(y1, y2 + 1))
        match op:
            case "turn on":
                grid |= BruteForceGrid.from_basic_grid(other)
            case "turn off":
                grid -= BruteForceGrid.from_basic_grid(other)
            case "toggle":
                grid ^= BruteForceGrid.from_basic_grid(other)
    return len(grid)


class BruteForceGrid(set[tuple[int, int]]):
    @classmethod
    def from_basic_grid(cls, grid: BasicGrid) -> BruteForceGrid:
        points = [(x, y) for x in grid.xs for y in grid.ys]
        return BruteForceGrid(points)


def fs(*items: T) -> frozenset[T]:
    return frozenset(items)


class Grid:
    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def __and__(self, other: Grid) -> Grid:
        ...

    def __or__(self, other: Grid) -> Grid:
        if other is NullGrid:
            return self
        return UnionGrid(fs(self, other))

    def __xor__(self, other: Grid) -> Grid:
        left = self - other
        right = other - self
        return left | right

    def __sub__(self, other: Grid) -> Grid:
        if other is NullGrid:
            return self
        return DifferenceGrid(self, other)


@dataclass(frozen=True)
class _NullGrid(Grid):
    _instance: ClassVar[Optional[_NullGrid]] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def count(self) -> int:
        return 0

    def __and__(self, other: Grid) -> Grid:
        return self._instance

    def __or__(self, other: Grid) -> Grid:
        return other

    def __xor__(self, other: Grid) -> Grid:
        return other

    def __sub__(self, other: Grid) -> Grid:
        return self._instance


NullGrid = _NullGrid()


@dataclass(frozen=True)
class BasicGrid(Grid):
    xs: range
    ys: range

    def count(self) -> int:
        return len(self.xs) * len(self.ys)

    def __and__(self, other: Grid) -> Grid:
        if isinstance(other, BasicGrid):
            xs = intersect_ranges(self.xs, other.xs)
            ys = intersect_ranges(self.ys, other.ys)
            if len(xs) == 0 or len(ys) == 0:
                return NullGrid
            return BasicGrid(xs, ys)
        return other & self

    def __or__(self, other: Grid) -> Grid:
        if not isinstance(other, BasicGrid):
            return other | self
        if other in self:
            return self
        if self in other:
            return other
        return UnionGrid(fs(self, other))

    def __contains__(self, item: BasicGrid) -> bool:
        return self & item == item

    def __sub__(self, other: Grid) -> Grid:
        if isinstance(other, BasicGrid):
            if self in other:
                return NullGrid
            if self & other is NullGrid:
                return self
        return super().__sub__(other)


def intersect_ranges(left: range, right: range) -> range:
    lower_bound = max(min(left), min(right))
    upper_bound = min(max(left), max(right))
    return range(lower_bound, upper_bound + 1)


@dataclass(frozen=True)
class UnionGrid(Grid):
    contents: frozenset[Grid]

    def count(self) -> int:
        result = sum(g.count() for g in self.contents)
        # TODO: subtract overlaps
        return result

    def __or__(self, other: Grid) -> Grid:
        if isinstance(other, UnionGrid):
            return UnionGrid(self.contents | other.contents)
        return UnionGrid(self.contents | {other})

    def __and__(self, other: Grid) -> Grid:
        return UnionGrid(frozenset(g & other for g in self.contents))

    def __sub__(self, other: Grid) -> Grid:
        return UnionGrid(frozenset(g - other for g in self.contents))


@dataclass(frozen=True)
class DifferenceGrid(Grid):
    minuend: Grid
    subtrahend: Grid

    def count(self) -> int:
        intersection = self.minuend & self.subtrahend
        return self.minuend.count() - intersection.count()

    def __and__(self, other: Grid) -> Grid:
        return (self.minuend & other) - self.subtrahend


SAMPLE_INPUTS = [
    """\
turn on 0,0 through 999,999
toggle 0,0 through 999,0
turn off 499,499 through 500,500
""",
    """\
turn on 0,0 through 1,1
turn on 1,1 through 2,2
turn on 1,0 through 3,1
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 998996), (1, 10)], indirect=["sample_input"]
)
def test_count_on_at_end(sample_input, expected):
    assert count_on_at_end(sample_input) == expected


UNIT_GRID = BasicGrid(range(0, 1), range(0, 1))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            BasicGrid(range(0, 1), range(0, 2)),
            BasicGrid(range(1, 2), range(0, 2)),
            UnionGrid(fs(BasicGrid(range(0, 1), range(0, 2)), BasicGrid(range(1, 2), range(0, 2)))),
        ),
        (UNIT_GRID, NullGrid, UNIT_GRID),
        (UNIT_GRID, UNIT_GRID, UNIT_GRID),
    ],
)
def test_union(left, right, expected):
    assert left | right == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (UNIT_GRID, UNIT_GRID, NullGrid),
        (UNIT_GRID, NullGrid, UNIT_GRID),
    ],
)
def test_xor(left, right, expected):
    assert left ^ right == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (UNIT_GRID, UNIT_GRID, UNIT_GRID),
        (UNIT_GRID, BasicGrid(range(0, 2), range(0, 2)), UNIT_GRID),
        (BasicGrid(range(-1, 1), range(-1, 1)), BasicGrid(range(0, 2), range(0, 2)), UNIT_GRID),
    ],
)
def test_intersect(left, right, expected):
    assert left & right == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            BasicGrid(range(0, 2), range(0, 2)),
            BasicGrid(range(1, 3), range(0, 2)),
            BasicGrid(range(0, 1), range(0, 2)),
        ),
        (
            BasicGrid(range(0, 5), range(0, 5)),
            BasicGrid(range(0, 5), range(1, 4)),
            UnionGrid(fs(BasicGrid(range(0, 1), range(0, 5)), BasicGrid(range(4, 5), range(0, 5)))),
        ),
    ],
)
def test_subtract(left, right, expected):
    assert left - right == expected
