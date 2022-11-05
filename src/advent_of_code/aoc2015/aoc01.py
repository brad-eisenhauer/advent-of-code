"""Advent of Code 2015, day 1: https://adventofcode.com/2015/day/1"""
from collections import Counter
from typing import Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            instructions = f.read().strip()
        return calc_final_floor(instructions)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            instructions = f.read().strip()
        return calc_first_basement_entry(instructions)


def calc_final_floor(instructions: str) -> int:
    c = Counter(instructions)
    return c["("] - c[")"]


def calc_first_basement_entry(instructions: str) -> Optional[int]:
    floor = 0
    for i, char in enumerate(instructions):
        match char:
            case "(":
                floor += 1
            case ")":
                if floor == 0:
                    return i + 1
                floor -= 1
    return None


@pytest.mark.parametrize(
    ("instructions", "expected"),
    [
        ("(())", 0),
        ("()()", 0),
        ("(((", 3),
        ("(()(()(", 3),
        ("))(((((", 3),
        ("())", -1),
        ("))(", -1),
        (")))", -3),
        (")())())", -3),
    ],
)
def test_calc_final_floor(instructions, expected):
    assert calc_final_floor(instructions) == expected


@pytest.mark.parametrize(("instructions", "expected"), [(")", 1), ("()())", 5)])
def test_calc_first_basement_entry(instructions, expected):
    assert calc_first_basement_entry(instructions) == expected
