"""Advent of Code 2022, day 15: https://adventofcode.com/2022/day/15"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from io import StringIO
from typing import ClassVar, Iterable, Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            sensor_array = [Sensor.parse(line) for line in f]
        covered: set[Vector] = set()
        for sensor in sensor_array:
            covered |= set(sensor.covers_in_row(2_000_000))
        for sensor in sensor_array:
            covered -= {sensor.nearest_beacon}
        return len(covered)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            sensor_array = [Sensor.parse(line) for line in f]
        return find_uncovered(sensor_array, range(4_000_001))


Vector: tuple[int, ...]


@dataclass
class Sensor:
    loc: Vector
    nearest_beacon: Vector

    VECTOR_PATTERN: ClassVar[re.Pattern] = re.compile(r"x=(-?\d+), y=(-?\d+)")

    @classmethod
    def parse(cls, text: str) -> Sensor:
        sensor_match = cls.VECTOR_PATTERN.search(text)
        beacon_match = cls.VECTOR_PATTERN.search(text, sensor_match.end() + 1)
        sensor_loc = tuple(int(n) for n in sensor_match.groups())
        beacon_loc = tuple(int(n) for n in beacon_match.groups())
        return Sensor(sensor_loc, beacon_loc)

    @cached_property
    def beacon_distance(self) -> int:
        return sum(abs(a - b) for a, b in zip(self.loc, self.nearest_beacon))

    def covers_in_row(self, y: int) -> Iterator[Vector]:
        sensor_x, sensor_y = self.loc
        max_offset = self.beacon_distance - abs(y - sensor_y)
        for x in range(sensor_x - max_offset, sensor_x + max_offset + 1):
            yield x, y

    def covers(self, loc: Vector) -> bool:
        dist = sum(abs(a - b) for a, b in zip(loc, self.loc))
        return dist <= self.beacon_distance

    def boundary(self) -> Iterator[Vector]:
        d = self.beacon_distance + 1
        loc_x, loc_y = self.loc
        for x, y in zip(range(loc_x, loc_x + d), range(loc_y + d, loc_y, -1)):
            yield x, y
        for x, y in zip(range(loc_x + d, loc_x, -1), range(loc_y, loc_y - d, -1)):
            yield x, y
        for x, y in zip(range(loc_x, loc_x - d, -1), range(loc_y - d, loc_y)):
            yield x, y
        for x, y in zip(range(loc_x - d, loc_x), range(loc_y, loc_y + d)):
            yield x, y


def find_uncovered(sensor_array: list[Sensor], bounds: range) -> Vector:
    for sensor in sensor_array:
        for loc in sensor.boundary():
            x, y = loc
            if x not in bounds or y not in bounds:
                continue
            if not any(s.covers(loc) for s in sensor_array):
                return 4_000_000 * x + y


SAMPLE_INPUTS = [
    """\
Sensor at x=2, y=18: closest beacon is at x=-2, y=15
Sensor at x=9, y=16: closest beacon is at x=10, y=16
Sensor at x=13, y=2: closest beacon is at x=15, y=3
Sensor at x=12, y=14: closest beacon is at x=10, y=16
Sensor at x=10, y=20: closest beacon is at x=10, y=16
Sensor at x=14, y=17: closest beacon is at x=10, y=16
Sensor at x=8, y=7: closest beacon is at x=2, y=10
Sensor at x=2, y=0: closest beacon is at x=2, y=10
Sensor at x=0, y=11: closest beacon is at x=2, y=10
Sensor at x=20, y=14: closest beacon is at x=25, y=17
Sensor at x=17, y=20: closest beacon is at x=21, y=22
Sensor at x=16, y=7: closest beacon is at x=15, y=3
Sensor at x=14, y=3: closest beacon is at x=15, y=3
Sensor at x=20, y=1: closest beacon is at x=15, y=3
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture()
def sensor_array(sample_input):
    return list(Sensor.parse(line) for line in sample_input)


def test_sensor_covers_in_row(sensor_array):
    covered: set[Vector] = set()
    for sensor in sensor_array:
        sensor_covers = set(sensor.covers_in_row(10))
        covered |= sensor_covers
    for sensor in sensor_array:
        covered -= {sensor.nearest_beacon}
    assert len(covered) == 26


def test_find_uncovered(sensor_array):
    assert find_uncovered(sensor_array, range(0, 21)) == 56_000_011
