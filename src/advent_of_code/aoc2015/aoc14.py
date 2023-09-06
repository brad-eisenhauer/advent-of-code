"""Advent of Code 2015, day 14: https://adventofcode.com/2015/day/14"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            reindeer = [Reindeer.from_str(line) for line in f]
        return max(r.calc_distance(2503) for r in reindeer)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            reindeer = [Reindeer.from_str(line) for line in f]
        race_result = race(reindeer, 2503)
        return max(race_result.values())


PATTERN = (
    r"^(?P<name>\w+) can fly "  # Reindeer name
    r"(?P<speed>\d+) km/s for "  # speed
    r"(?P<travel_time>\d+) seconds, but then must rest for "  # movement period
    r"(?P<rest_time>\d+) seconds\.$"
)


@dataclass(frozen=True)
class Reindeer:
    name: str
    speed: int
    travel_time: int
    rest_time: int

    @classmethod
    def from_str(cls, text: str) -> Reindeer:
        match_ = re.match(PATTERN, text)
        if match_ is None:
            raise ValueError(f"Malformed description: '{text}'")
        props = match_.groupdict()
        for key in ["speed", "travel_time", "rest_time"]:
            props[key] = int(props[key])
        return Reindeer(**props)

    def calc_distance(self, time: int) -> int:
        cycle_time = self.travel_time + self.rest_time
        result = self.speed * self.travel_time * (time // cycle_time)
        result += min(self.travel_time, time % cycle_time) * self.speed
        return result


def race(reindeer: list[Reindeer], time: int) -> dict[str, int]:
    result = defaultdict(int)
    max_dist = 0
    for t in range(1, time + 1):
        leaders = []
        for r in reindeer:
            d = r.calc_distance(t)
            if d > max_dist:
                leaders = [r]
                max_dist = d
            elif d == max_dist:
                leaders.append(r)
        for r in leaders:
            result[r.name] += 1
    return result


def load_reindeer(f: IO) -> Iterator[Reindeer]:
    for line in f:
        yield Reindeer.from_str(line)


SAMPLE_INPUTS = [
    """\
Comet can fly 14 km/s for 10 seconds, but then must rest for 127 seconds.
Dancer can fly 16 km/s for 11 seconds, but then must rest for 162 seconds.
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture
def reindeer():
    return [Reindeer("Comet", 14, 10, 127), Reindeer("Dancer", 16, 11, 162)]



def test_load_reindeer(sample_input, reindeer):
    reindeer = list(load_reindeer(sample_input))
    assert reindeer == reindeer


@pytest.mark.parametrize(
    ("reindeer", "time", "expected"),
    [
        (Reindeer("Comet", 14, 10, 127), 1000, 1120),
        (Reindeer("Dancer", 16, 11, 162), 1000, 1056),
    ]
)
def test_calc_distance(reindeer, time, expected):
    assert reindeer.calc_distance(time) == expected


def test_race(reindeer):
    result = race(reindeer, 1000)
    assert result == {"Comet": 312, "Dancer": 689}
