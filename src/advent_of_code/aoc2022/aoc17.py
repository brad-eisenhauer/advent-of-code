"""Advent of Code 2022, day 17: https://adventofcode.com/2022/day/17"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from io import StringIO
from itertools import chain, repeat, takewhile
from typing import ClassVar, Iterable, Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            directions = f.readline().rstrip()
        return calc_height(2022, directions)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            directions = f.readline().rstrip()
        return calc_height(1_000_000_000_000, directions)


Vector = tuple[int, ...]

UNIT_VECTORS = {"<": (-1, 0), ">": (1, 0), "v": (0, -1)}
SHAPES = """\
####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##
"""


@dataclass(frozen=True)
class Shape:
    locs: frozenset[Vector]
    _empty_instance: ClassVar[Shape] = None

    @classmethod
    def parse(cls, f: TextIO) -> Optional[Shape]:
        locs = []
        for y, line in enumerate(takewhile(lambda s: s.rstrip() != "", f)):
            for x, char in enumerate(line):
                if char == "#":
                    locs.append((x, y))
        if not locs:
            return None
        max_y = max(v[1] for v in locs)
        return Shape.from_iterable((x, max_y - y) for x, y in locs)

    @classmethod
    def from_iterable(cls, locs: Iterable[Vector]) -> Shape:
        return cls(frozenset(locs))

    @classmethod
    def empty(cls) -> Shape:
        if cls._empty_instance is None:
            cls._empty_instance = Shape(frozenset())
        return cls._empty_instance

    def translate(self, offset: Vector) -> Shape:
        return Shape.from_iterable(tuple(a + b for a, b in zip(offset, loc)) for loc in self.locs)

    def intersects(self, other: Shape) -> bool:
        return any(v in other.locs for v in self.locs)

    def contains(self, other: Shape) -> bool:
        return all(v in self.locs for v in other.locs)

    def extend(self, locs: Iterable[Vector]) -> Shape:
        return Shape.from_iterable(chain(self.locs, locs))

    def height(self) -> int:
        return max(y for _, y in self.locs)

    def __sub__(self, other: Shape) -> Shape:
        return Shape(self.locs - other.locs)

    def trim(self) -> Shape:
        frontier = deque([(3, self.height())])
        result = set()
        while frontier:
            loc = frontier.popleft()
            if loc in result:
                continue
            result.add(loc)
            x, y = loc
            frontier.extend(
                (x + dx, y + dy)
                for dx, dy in UNIT_VECTORS.values()
                if (x + dx, y + dy) in self.locs
            )
        return Shape.from_iterable(result)


def parse_all_shapes(f: TextIO) -> list[Shape]:
    result = []
    while True:
        shape = Shape.parse(f)
        if shape is None:
            break
        result.append(shape)
    return result


def drop_shape(
    boundary: Shape, shape: Shape, directions: Iterator[str], height: int
) -> tuple[Shape, Shape, int]:
    """Drop a single shape

    Parameters
    ----------
    boundary
    shape
    directions
    height

    Returns
    -------
    Dropped shape at rest, resulting boundary, and number of directions consumed
    """
    shape = shape.translate((2, height + 4))
    boundary = boundary.extend(
        (x, y) for x in range(7) for y in range(height + 1, shape.height() + 1)
    )
    directions_consumed = 0
    while True:
        # push
        direction = next(directions)
        directions_consumed += 1
        next_shape = shape.translate(UNIT_VECTORS[direction])
        if boundary.contains(next_shape):
            shape = next_shape
        # fall
        next_shape = shape.translate(UNIT_VECTORS["v"])
        if boundary.contains(next_shape):
            shape = next_shape
        else:
            return shape, boundary - shape, directions_consumed


@dataclass(frozen=True)
class State:
    shape_index: int
    direction_index: int
    relative_height_history: tuple[int, ...]
    height: int = field(compare=False)


HISTORY_COMPARISON_LENGTH = 50


def find_cycle(
    boundary: Shape, shapes: list[Shape], directions: str
) -> tuple[list[State], int, int]:
    direction_count = len(directions)
    shape_count = len(shapes)
    states: list[State] = [State(0, 0, (), 0)]
    last_state = states[0]

    shapes = chain.from_iterable(repeat(shapes))
    directions = chain.from_iterable(repeat(directions))

    while True:
        shape, boundary, d_count = drop_shape(boundary, next(shapes), directions, last_state.height)
        new_height = max(last_state.height, shape.height())
        states.append(
            State(
                shape_index=(last_state.shape_index + 1) % shape_count,
                direction_index=(last_state.direction_index + d_count) % direction_count,
                relative_height_history=(
                    new_height - last_state.height,
                    *last_state.relative_height_history[: HISTORY_COMPARISON_LENGTH - 1],
                ),
                height=new_height,
            ),
        )
        p2 = len(states) - 1
        p1 = p2 // 2

        if states[p1] == states[p2]:
            return states, p1, p2

        if p2 % 50 == 49:
            boundary = boundary.trim()

        last_state = states[-1]


def calc_height(n: int, directions: str) -> int:
    shapes = parse_all_shapes(StringIO(SHAPES))
    states, p1, p2 = find_cycle(Shape.empty(), shapes, directions)
    shapes_remaining = n - p2

    if shapes_remaining <= 0:
        return states[n].height

    cycles_remaining = shapes_remaining // (p2 - p1)
    remainder = shapes_remaining % (p2 - p1)
    result = states[p2].height
    result += cycles_remaining * (states[p2].height - states[p1].height)
    result += states[p1 + remainder].height - states[p1].height
    return result


SAMPLE_INPUTS = [
    """\
>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_parse_all_shapes():
    f = StringIO(SHAPES)
    shapes = parse_all_shapes(f)
    assert len(shapes) == 5
    assert shapes[2] == Shape.from_iterable([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])


@pytest.mark.parametrize(("count", "expected"), [(2022, 3068), (1_000_000_000_000, 1514285714288)])
def test_calc_height(sample_input, count, expected):
    directions = sample_input.readline().rstrip()
    assert calc_height(count, directions) == expected
