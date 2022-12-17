"""Advent of Code 2022, day 17: https://adventofcode.com/2022/day/17"""
from __future__ import annotations

from collections import deque
from io import StringIO
from itertools import chain, repeat, takewhile
from typing import Iterable, Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            directions = f.readline().rstrip()
        shapes = parse_all_shapes(StringIO(SHAPES))
        return max(
            shape.height()
            for shape in drop_shapes(
                Shape.empty(),
                2022,
                chain.from_iterable(repeat(shapes)),
                chain.from_iterable(repeat(directions)),
            )
        )


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


class Shape:
    _empty_instance: Shape = None

    def __init__(self, spaces: Iterable[Vector]):
        self._locs = frozenset(spaces)

    def __eq__(self, other):
        return isinstance(other, Shape) and self._locs == other._locs

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
        return Shape((x, max_y - y) for x, y in locs)

    @classmethod
    def empty(cls) -> Shape:
        if cls._empty_instance is None:
            cls._empty_instance = Shape(())
        return cls._empty_instance

    @property
    def locs(self) -> Iterable[Vector]:
        return self._locs

    def translate(self, offset: Vector) -> Shape:
        return Shape(tuple(a + b for a, b in zip(offset, loc)) for loc in self._locs)

    def intersects(self, other: Shape) -> bool:
        return any(v in other._locs for v in self._locs)

    def contains(self, other: Shape) -> bool:
        return all(v in self._locs for v in other._locs)

    def extend(self, locs: Iterable[Vector]) -> Shape:
        return Shape(chain(self._locs, locs))

    def height(self) -> int:
        return max(y for _, y in self._locs)

    def __sub__(self, other: Shape) -> Shape:
        return Shape(self._locs - other._locs)

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
                if (x + dx, y + dy) in self._locs
            )
        return Shape(result)


def parse_all_shapes(f: TextIO) -> list[Shape]:
    result = []
    while True:
        shape = Shape.parse(f)
        if shape is None:
            break
        result.append(shape)
    return result


def drop_shapes(
    boundary: Shape, n: int, shapes: Iterator[Shape], directions: Iterator[str]
) -> Iterator[Shape]:
    height = 0
    for i in range(n):
        if i % 50 == 49:
            boundary = boundary.trim()
        shape = next(shapes).translate((2, height + 4))
        boundary = boundary.extend(
            (x, y) for x in range(7) for y in range(height + 1, shape.height() + 1)
        )
        while True:
            # push
            direction = next(directions)
            next_shape = shape.translate(UNIT_VECTORS[direction])
            if boundary.contains(next_shape):
                shape = next_shape
            # fall
            next_shape = shape.translate(UNIT_VECTORS["v"])
            if boundary.contains(next_shape):
                shape = next_shape
            else:  # shape is at rest
                yield shape
                height = max(height, shape.height())
                boundary -= shape
                break


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
    boundary = Shape.empty()
    result = max(
        s.height()
        for s in drop_shapes(
            boundary,
            9,
            chain.from_iterable(repeat(shapes)),
            chain.from_iterable(repeat(directions)),
        )
    )
    assert result == 17


def test_parse_all_shapes():
    f = StringIO(SHAPES)
    shapes = parse_all_shapes(f)
    assert len(shapes) == 5
    assert shapes[2] == Shape([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])
