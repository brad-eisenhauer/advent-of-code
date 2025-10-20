"""Advent of Code 2023, day 1: https://adventofcode.com/2023/day/1"""

from __future__ import annotations

import re
from functools import cache
from io import StringIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2023, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            return sum(int(find_first_and_last_digits(line, digits_only=True)) for line in fp)

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            return sum(int(find_first_and_last_digits(line)) for line in fp)


DIGIT_STRINGS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def find_first_and_last_digits(text: str, digits_only: bool = False) -> str:
    pattern = _build_pattern(digits_only)
    first_match, last_match = re.search(pattern, text).groups()
    first_digit = _coerce_digit(first_match)
    if last_match is not None:
        last_digit = _coerce_digit(last_match or first_match)
    elif digits_only:
        last_digit = first_digit
    else:
        match = re.search(_build_pattern(digits_only, reversed=True), text[::-1])
        last_digit = _coerce_digit(match.group(1)[::-1])
    return first_digit + last_digit


def _coerce_digit(t: str) -> str:
    return DIGIT_STRINGS.get(t, t)


@cache
def _build_pattern(digits_only: bool, build_reversed: bool = False) -> str:
    if digits_only:
        digits_pattern = r"(\d)"
    elif build_reversed:
        digits_pattern = rf"(\d|{'|'.join(s[::-1] for s in DIGIT_STRINGS)})"
    else:
        digits_pattern = rf"(\d|{'|'.join(DIGIT_STRINGS.keys())})"
    return rf"{digits_pattern}(?:.*{digits_pattern})?"


SAMPLE_INPUTS = [
    """\
1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet
""",
    """\
two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen
""",
    """\
sevenine
eightwo
oneight
threeightwo
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "digits_only", "expected"),
    [
        (0, True, ["12", "38", "15", "77"]),
        (1, False, ["29", "83", "13", "24", "42", "14", "76"]),
        (2, False, ["79", "82", "18", "32"]),
    ],
    indirect=["sample_input"],
)
def test_find_first_and_last_digits(sample_input, digits_only, expected):
    assert [find_first_and_last_digits(line, digits_only) for line in sample_input] == expected
