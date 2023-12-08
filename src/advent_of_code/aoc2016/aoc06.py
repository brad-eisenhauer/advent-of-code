"""Advent of Code 2016, day 6: https://adventofcode.com/2016/day/6"""
from __future__ import annotations

from collections import Counter
from io import StringIO
from typing import IO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            return recover_message(fp)

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            return recover_message(fp, modified=True)


def recover_message(messages: IO, modified: bool = False) -> str:
    char_counts = [Counter(chars) for chars in zip(*(message.strip() for message in messages))]
    recovered_chars = [
        sorted(counter.items(), key=lambda c: c[1], reverse=not modified)[0][0]
        for counter in char_counts
    ]
    return "".join(recovered_chars)


SAMPLE_INPUTS = [
    """\
eedadn
drvtee
eandsr
raavrd
atevrs
tsrnev
sdttsa
rasrtv
nssdts
ntnada
svetve
tesnvt
vntsnd
vrdear
dvrsen
enarar
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(("modified", "expected"), [(False, "easter"), (True, "advent")])
def test_recover_message(sample_input, modified, expected):
    assert recover_message(sample_input, modified) == expected
