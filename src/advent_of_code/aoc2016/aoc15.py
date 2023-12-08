"""Advent of Code 2016, day 15: https://adventofcode.com/2016/day/15"""
from __future__ import annotations
from dataclasses import dataclass

from io import StringIO
from itertools import count
import re
from typing import IO, Iterable, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.math import chinese_remainder_theorem


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            discs = [Disc.read(line) for line in fp]
        divisors = [d.period for d in discs]
        remainders = [(-d.start_index - d.id) % d.period for d in discs]
        return chinese_remainder_theorem(divisors, remainders)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            discs = [Disc.read(line) for line in fp]
        discs.append(Disc(len(discs) + 1, 11, 0))
        log.debug("%s", discs)
        divisors = [d.period for d in discs]
        remainders = [(-d.start_index - d.id) % d.period for d in discs]
        return chinese_remainder_theorem(divisors, remainders)


@dataclass(frozen=True)
class Disc:
    id: int
    period: int
    start_index: int

    @classmethod
    def read(cls, text: str) -> Disc:
        pattern = r"Disc #(\d+) has (\d+) positions;.*at position (\d+)\."
        id_str, period_str, start_str = re.match(pattern, text).groups()
        return cls(int(id_str), int(period_str), int(start_str))


SAMPLE_INPUTS = [
    """\
Disc #1 has 5 positions; at time=0, it is at position 4.
Disc #2 has 2 positions; at time=0, it is at position 1.
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
    assert solution.solve_part_one(sample_input) == 5


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 85

def test_disc_read(sample_input):
    result = [Disc.read(line) for line in sample_input]
    assert result == [Disc(1, 5, 4), Disc(2, 2, 1)]


@pytest.mark.parametrize(
    ("divisors", "remainders", "expected"),
    [
        ([3, 5, 7], [2, 3, 2], 23),
        ([3, 4, 5], [0, 3, 4], 39),
        ([5, 2], [0, 1], 5),
    ]
)
def test_chinese_remainder_theorem(divisors, remainders, expected):
    assert chinese_remainder_theorem(divisors, remainders) == expected
