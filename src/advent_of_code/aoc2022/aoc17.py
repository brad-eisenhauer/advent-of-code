"""Advent of Code 2022, day 17: https://adventofcode.com/2022/day/17"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache
from io import StringIO
from itertools import chain, islice, repeat, takewhile
from typing import ClassVar, Iterable, Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            directions = f.readline().rstrip()
        shapes = parse_all_shapes(StringIO(SHAPES))
        result, max_fall = drop_shapes(
            Shape.empty(),
            2022,
            chain.from_iterable(repeat(shapes)),
            chain.from_iterable(repeat(directions)),
        )
        print(f"Max fall was {max_fall}.")
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            directions = f.readline().rstrip()
        shapes = parse_all_shapes(StringIO(SHAPES))
        result, max_fall = drop_shapes(
            Shape.empty(),
            int(1e12),
            chain.from_iterable(repeat(shapes)),
            chain.from_iterable(repeat(directions)),
        )
        print(f"Max fall was {max_fall}.")
        return result


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
    boundary: Shape, shape: Shape, directions: tuple[str, ...]
) -> tuple[Shape, Shape, int]:
    """

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
    shape = shape.translate((2, 4))
    boundary = boundary.extend((x, y) for x in range(7) for y in range(1, shape.height() + 1))
    directions_consumed = 0
    while True:
        # push
        direction, *directions = directions
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


def drop_shapes(
    boundary: Shape,
    n: int,
    shapes: Iterator[Shape],
    directions: Iterator[str],
    chunk_size: int = 36,
) -> tuple[int, int]:
    height = 0
    max_fall = 0
    chunked_directions: tuple[str, ...] = ()
    memoized_drop_shape = cache(drop_shape)
    for i in range(n):
        chunked_directions = (
            *chunked_directions,
            *islice(directions, chunk_size - len(chunked_directions)),
        )
        next_shape = next(shapes)
        start_height = next_shape.height()
        shape, boundary, d_count = memoized_drop_shape(boundary, next_shape, chunked_directions)
        fall = start_height - shape.height() + 4
        max_fall = max(max_fall, fall)
        delta_height = shape.height()
        chunked_directions = chunked_directions[d_count:]
        boundary = boundary.trim()
        if delta_height > 0:
            boundary = boundary.translate((0, -delta_height))
            height += delta_height
    return height, max_fall


SAMPLE_INPUTS = [
    """\
>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_drop_shapes(sample_input):
    directions = sample_input.readline().rstrip()
    shapes = parse_all_shapes(StringIO(SHAPES))
    result, _ = drop_shapes(
        Shape.empty(),
        9,
        chain.from_iterable(repeat(shapes)),
        chain.from_iterable(repeat(directions)),
    )
    assert result == 17


def test_parse_all_shapes():
    f = StringIO(SHAPES)
    shapes = parse_all_shapes(f)
    assert len(shapes) == 5
    assert shapes[2] == Shape.from_iterable([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])
