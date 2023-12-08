"""Advent of Code 2023, day 6: https://adventofcode.com/2023/day/6"""
from __future__ import annotations

import math
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import product


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            times, dists = read_input(fp)
        return product(count_ways_to_win(t, d) for t, d in zip(times, dists))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            time, dist = read_input_2(fp)
        return count_ways_to_win(time, dist)


def read_input(fp: IO) -> tuple[list[int], list[int]]:
    def _parser(text: str):
        return [int(n) for n in text.split()]

    _, time_str = fp.readline().split(":")
    _, dist_str = fp.readline().split(":")
    return _parser(time_str), _parser(dist_str)


def read_input_2(fp: IO) -> tuple[int, int]:
    def _parser(text: str):
        return int(text.replace(" ", ""))

    _, time_str = fp.readline().split(":")
    _, dist_str = fp.readline().split(":")
    return _parser(time_str), _parser(dist_str)


def calc_dist(time: int, prep: int) -> int:
    if prep < 0 or time < prep:
        raise ValueError(f"Prep time must be in range 0 to {time}.")
    return prep * (time - prep)


def calc_prep(time: int, dist: int) -> Optional[int]:
    """Calculate the minimum prep time, if any, to travel at least dist.

    dist == -(prep ** 2) + time * prep
    -(prep ** 2) + time * prep - dist == 0
    """
    a = -1
    b = time
    c = -dist
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    return int(math.ceil((-b + math.sqrt(discriminant)) / (2 * a)))


def count_ways_to_win(time: int, dist: int) -> Optional[int]:
    min_prep = calc_prep(time, dist + 1)
    if min_prep is None:
        return None
    max_prep = time - min_prep
    result = max_prep - min_prep + 1
    return result


SAMPLE_INPUTS = [
    """\
Time:      7  15   30
Distance:  9  40  200
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 288


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 71503


def test_read_input(sample_input):
    assert read_input(sample_input) == ([7, 15, 30], [9, 40, 200])


@pytest.mark.parametrize(
    ("time", "prep", "expected"),
    [
        (7, 0, 0),
        (7, 1, 6),
        (7, 2, 10),
        (7, 3, 12),
        (7, 4, 12),
        (7, 5, 10),
        (7, 6, 6),
        (7, 7, 0),
    ],
)
def test_calc_dist(time, prep, expected):
    assert calc_dist(time, prep) == expected


@pytest.mark.parametrize(
    ("time", "dist", "expected"),
    [
        (7, 9, 2),
        (15, 40, 4),
        (30, 200, 10),
    ],
)
def test_calc_prep(time, dist, expected):
    assert calc_prep(time, dist) == expected


@pytest.mark.parametrize(
    ("time", "dist", "expected"),
    [
        (7, 9, 4),
        (15, 40, 8),
        (30, 200, 9),
    ],
)
def test_count_ways_to_win(time, dist, expected):
    assert count_ways_to_win(time, dist) == expected
