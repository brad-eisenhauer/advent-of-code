"""Advent of Code 2020, day 6: https://adventofcode.com/2020/day/6"""

import operator
from functools import reduce
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return sum(count_any_yes_answers_by_group(f))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return sum(count_all_yes_answers_by_group(f))


def count_any_yes_answers_by_group(f: TextIO) -> Iterator[int]:
    for group in f.read().split("\n\n"):
        yield len(set(group) - {"\n"})


def count_all_yes_answers_by_group(f: TextIO) -> Iterator[int]:
    for group in f.read().split("\n\n"):
        common_answers = reduce(operator.and_, (set(line) for line in group.split()))
        yield len(common_answers)


SAMPLE_INPUT = """\
abc

a
b
c

ab
ac

a
a
a
a

b
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_count_any_yes_answers_by_group(sample_input):
    expected = [3, 3, 3, 1, 1]
    result = list(count_any_yes_answers_by_group(sample_input))
    assert result == expected


def test_count_all_yes_answers_by_group(sample_input):
    expected = [3, 0, 1, 1, 1]
    result = list(count_all_yes_answers_by_group(sample_input))
    assert result == expected
