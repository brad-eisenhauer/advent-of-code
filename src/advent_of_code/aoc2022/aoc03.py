"""Advent of Code 2022, day 3: https://adventofcode.com/2022/day/3"""
from __future__ import annotations

import operator
from functools import reduce
from io import StringIO
from itertools import islice
from typing import Iterable, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return sum_misplaced_priorities(f)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return sum_badge_type_priorities(f)


def find_misplaced_item(text: str) -> str:
    left_items = set(text[: len(text) // 2])
    right_items = set(text[len(text) // 2 :])
    return next(iter(left_items & right_items))


def calc_item_priority(text: str) -> int:
    if text.islower():
        return ord(text) - ord("a") + 1
    if text.isupper():
        return ord(text) - ord("A") + 27


def sum_misplaced_priorities(f: TextIO) -> int:
    result = 0
    for line in f:
        misplaced_item = find_misplaced_item(line.strip())
        result += calc_item_priority(misplaced_item)
    return result


def find_badge_type(contents: Iterable[str]) -> str:
    content_types = (set(content) for content in contents)
    common_content = reduce(operator.and_, content_types)
    return next(iter(common_content))


def sum_badge_type_priorities(f: TextIO) -> int:
    result = 0
    while True:
        contents = [line.strip() for line in islice(f, 3)]
        if not contents:
            break
        badge_type = find_badge_type(contents)
        result += calc_item_priority(badge_type)
    return result


SAMPLE_INPUTS = [
    """\
vJrwpWtwJgWrhcsFMMfFFhFp
jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL
PmmdzqPrVvPwwTWBwg
wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn
ttgJtRGJQctTZtZT
CrZsJsPPZsGzwwsLwLmpwMDw
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(
    ("line_index", "expected"), [(0, "p"), (1, "L"), (2, "P"), (3, "v"), (4, "t"), (5, "s")]
)
def test_find_misplaced_item(sample_input, line_index, expected):
    lines = sample_input.readlines()
    assert find_misplaced_item(lines[line_index]) == expected


@pytest.mark.parametrize(
    ("item", "priority"), [("p", 16), ("L", 38), ("P", 42), ("v", 22), ("t", 20), ("s", 19)]
)
def test_calc_item_priority(item, priority):
    assert calc_item_priority(item) == priority


def test_sum_misplaced_priorities(sample_input):
    assert sum_misplaced_priorities(sample_input) == 157


@pytest.mark.parametrize(("line_index", "expected"), [(0, "r"), (3, "Z")])
def test_find_badge_type(sample_input, line_index, expected):
    lines = [line.strip() for line in sample_input]
    assert find_badge_type(lines[line_index : line_index + 3]) == expected


def test_sum_badge_type_priorities(sample_input):
    assert sum_badge_type_priorities(sample_input) == 70
