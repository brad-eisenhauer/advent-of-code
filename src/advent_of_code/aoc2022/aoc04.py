"""Advent of Code 2022, day 4: https://adventofcode.com/2022/day/4"""
from __future__ import annotations

import re
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return count_either_contains(f)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return count_overlaps(f)


def count_either_contains(f: TextIO) -> int:
    result = 0
    for line in f:
        left, right = parse_line(line)
        intersection = intersect_ranges(left, right)
        if intersection == left or intersection == right:
            result += 1
    return result


def count_overlaps(f: TextIO) -> int:
    result = 0
    for line in f:
        left, right = parse_line(line)
        intersection = intersect_ranges(left, right)
        if len(intersection) > 0:
            result += 1
    return result


PATTERN = re.compile(r"(\d+)-(\d+),(\d+)-(\d+)")


def parse_line(line: str) -> tuple[range, range]:
    match = PATTERN.match(line)
    x1, x2, y1, y2 = (int(n) for n in match.groups())
    return range(x1, x2 + 1), range(y1, y2 + 1)


def intersect_ranges(left: range, right: range) -> range:
    int_min = max(min(left), min(right))
    int_max = min(max(left), max(right))
    return range(int_min, int_max + 1)


SAMPLE_INPUTS = [
    """\
2-4,6-8
2-3,4-5
5-7,7-9
2-8,3-7
6-6,4-6
2-6,4-8
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_count_either_contains(sample_input):
    assert count_either_contains(sample_input) == 2


def test_count_overlaps(sample_input):
    assert count_overlaps(sample_input) == 4
