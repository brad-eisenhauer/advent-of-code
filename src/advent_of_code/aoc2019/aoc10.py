"""Advent of Code 2019, Day 10: https://adventofcode.com/2019/day/10"""
from __future__ import annotations

import math
from dataclasses import dataclass
from io import StringIO
from itertools import islice
from typing import Iterable, Iterator, TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util import greatest_common_divisor


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(10, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            placer = StationPlacer(StationPlacer.read_map(f))
        max_visible, _ = placer.calc_max_visible()
        return max_visible

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            placer = StationPlacer(StationPlacer.read_map(f))
        _, station = placer.calc_max_visible()
        x, y = next(islice(placer.vaporize(station), 199, None))
        return 100 * x + y


class StationPlacer:
    def __init__(self, map: Iterable[Vector]):
        self.map = set(map)

    @staticmethod
    def read_map(f: TextIO) -> Iterator[Vector]:
        for row_idx, line in enumerate(f.readlines()):
            line = line.strip()
            for col_idx, char in enumerate(line):
                if char == "#":
                    yield Vector((col_idx, row_idx))

    def calc_max_visible(self) -> tuple[int, Vector]:
        station = None
        max_visible = 0
        for left in self.map:
            visible_count = sum(
                1 for right in self.map if right is not left and self.has_line_of_sight(left, right)
            )
            if station is None or visible_count > max_visible:
                station = left
                max_visible = visible_count
        return max_visible, station

    def has_line_of_sight(self, left: Vector, right: Vector) -> bool:
        diff = right - left
        for s in diff.get_steps():
            if left + s in self.map:
                return False
        return True

    def vaporize(self, station: Vector) -> Iterator[Vector]:
        self.map.remove(station)
        while self.map:
            for a in sorted(
                (a for a in self.map if self.has_line_of_sight(station, a)),
                key=lambda a: (a - station).heading,
            ):
                self.map.remove(a)
                yield a


@dataclass(frozen=True)
class Vector:
    vector: tuple[int, ...]

    @staticmethod
    def null(dim: int) -> Vector:
        return Vector((0,) * dim)

    def __len__(self) -> int:
        return len(self.vector)

    def __getitem__(self, item: int) -> int:
        return self.vector[item]

    def __iter__(self) -> Iterator[int]:
        return iter(self.vector)

    def __add__(self, other: Vector) -> Vector:
        assert len(other) == len(self)
        return Vector(tuple(a + b for a, b in zip(self, other)))

    def __sub__(self, other: Vector) -> Vector:
        assert len(other) == len(self)
        return Vector(tuple(a - b for a, b in zip(self, other)))

    def __mul__(self, other: int) -> Vector:
        return Vector(tuple(other * a for a in self))

    def get_steps(self) -> Iterator[Vector]:
        if any(self):
            div = greatest_common_divisor(*self.vector)
            step = Vector(tuple(n // div for n in self))
            for n in range(1, div):
                yield step * n

    @property
    def heading(self) -> float:
        x, y = self
        return math.degrees(math.atan2(x, -y)) % 360.0


class TestVector:
    @pytest.mark.parametrize(
        ("vector", "expected"),
        [
            (Vector.null(2), []),
            (Vector((15, 14)), []),
            (Vector((0, 3)), [Vector((0, 1)), Vector((0, 2))]),
            (Vector((4, 12)), [Vector((1, 3)), Vector((2, 6)), Vector((3, 9))]),
            (Vector((-2, -2)), [Vector((-1, -1))]),
        ],
    )
    def test_get_steps(self, vector, expected):
        assert list(vector.get_steps()) == expected

    def test_sub(self):
        assert Vector((2, 2)) - Vector((0, 2)) == Vector((2, 0))

    @pytest.mark.parametrize(
        ("vector", "expected"),
        [
            (Vector((0, -1)), 0.0),
            (Vector((1, -1)), 45.0),
            (Vector((1, 0)), 90.0),
            (Vector((-1, -1)), 315.0),
        ],
    )
    def test_heading(self, vector, expected):
        assert vector.heading == expected


SAMPLE_MAPS = [
    """\
.#..#
.....
#####
....#
...##
""",  # 8 visible from (3, 4)
    """\
......#.#.
#..#.#....
..#######.
.#.#.###..
.#..#.....
..#....#.#
#..#....#.
.##.#..###
##...#..#.
.#....####
""",  # 33 visible from (5, 8)
    """\
#.#...#.#.
.###....#.
.#....#...
##.#.#.#.#
....#.#.#.
.##..###.#
..#...##..
..##....##
......#...
.####.###.
""",  # 35 visible from (1, 2)
    """\
.#..#..###
####.###.#
....###.#.
..###.##.#
##.##.#.#.
....###..#
..#.#..#.#
#..#.#.###
.##...##.#
.....#.#..
""",  # 41 visible from (6, 3)
    """\
.#..##.###...#######
##.############..##.
.#.######.########.#
.###.#######.####.#.
#####.##.#.##.###.##
..#####..#.#########
####################
#.####....###.#.#.##
##.#################
#####.##.###..####..
..######..##.#######
####.##.####...##..#
.#####..#.######.###
##...#.##########...
#.##########.#######
.####.#.###.###.#.##
....##.##.###..#####
.#.#.###########.###
#.#.#.#####.####.###
###.##.####.##.#..##
""",  # 210 visible from (11, 13)
]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (Vector((0, 2)), Vector((1, 2)), True),
        (Vector((0, 2)), Vector((2, 2)), False),
    ],
)
def test_has_line_of_sight(left, right, expected):
    with StringIO(SAMPLE_MAPS[0]) as f:
        placer = StationPlacer(StationPlacer.read_map(f))
    assert placer.has_line_of_sight(left, right) == expected


@pytest.mark.parametrize(("map_input", "expected"), zip(SAMPLE_MAPS, [8, 33, 35, 41, 210]))
def test_calc_max_visible_asteroids(map_input, expected):
    with StringIO(map_input) as f:
        placer = StationPlacer(StationPlacer.read_map(f))
    max_visible, _ = placer.calc_max_visible()
    assert max_visible == expected


def test_vaporize():
    with StringIO(SAMPLE_MAPS[-1]) as f:
        placer = StationPlacer(StationPlacer.read_map(f))
    _, station = placer.calc_max_visible()
    assert station == Vector((11, 13))
    asteroids = list(placer.vaporize(station))
    assert asteroids[:3] == [Vector((11, 12)), Vector((12, 1)), Vector((12, 2))]
    assert asteroids[9] == Vector((12, 8))
    assert asteroids[199] == Vector((8, 2))
