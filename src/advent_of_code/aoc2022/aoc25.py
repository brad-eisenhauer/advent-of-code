"""Advent of Code 2022, day 25: https://adventofcode.com/2022/day/25"""
from __future__ import annotations

from io import StringIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[str, None]):
    def __init__(self, **kwargs):
        super().__init__(25, 2022, **kwargs)

    def solve_part_one(self) -> str:
        total = 0
        with self.open_input() as f:
            for snafu in f:
                total += to_int(snafu.rstrip())
        return to_snafu(total)


SNAFU_DIGITS = {0: "=", 1: "-", 2: "0", 3: "1", 4: "2"}
SNAFU_DIGITS_REVERSED = {sym: val for val, sym in SNAFU_DIGITS.items()}


def to_int(snafu: str) -> int:
    if not snafu:
        return 0
    result = SNAFU_DIGITS_REVERSED[snafu[-1]] - 2
    return result + 5 * to_int(snafu[:-1])


def to_snafu(n: int) -> str:
    ones = (n + 2) % 5
    fives = (n + 2) // 5
    result = SNAFU_DIGITS[ones]
    if fives > 0:
        result = to_snafu(fives) + result
    return result


SAMPLE_INPUTS = [
    """\
1=-0-2
12111
2=0=
21
2=01
111
20012
112
1=-1=
1-12
12
1=
122
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(
    ("line_index", "expected"),
    [
        (0, 1747),
        (1, 906),
        (2, 198),
        (3, 11),
        (4, 201),
        (5, 31),
        (6, 1257),
        (7, 32),
        (8, 353),
        (9, 107),
        (10, 7),
        (11, 3),
        (12, 37),
    ],
)
def test_to_int(sample_input, line_index, expected):
    lines = sample_input.readlines()
    assert to_int(lines[line_index].rstrip()) == expected


@pytest.mark.parametrize(
    ("line_index", "value"),
    [
        (0, 1747),
        (1, 906),
        (2, 198),
        (3, 11),
        (4, 201),
        (5, 31),
        (6, 1257),
        (7, 32),
        (8, 353),
        (9, 107),
        (10, 7),
        (11, 3),
        (12, 37),
    ],
)
def test_to_snafu(sample_input, line_index, value):
    lines = sample_input.readlines()
    assert to_snafu(value) == lines[line_index].rstrip()
