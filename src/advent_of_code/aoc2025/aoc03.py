"""Advent of Code 2025, day 3: https://adventofcode.com/2025/day/3"""

from __future__ import annotations

from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            return sum(calc_max_joltage(line.strip()) for line in fp)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            return sum(calc_max_joltage(line.strip(), n=12) for line in fp)


def calc_max_joltage(text: str, n: int = 2) -> int:
    digits = []
    while n > 1:
        if n == len(text):
            digits.append(text)
            break
        max_char = max(text[:1-n])
        max_index = text.index(max_char)
        digits.append(max_char)
        text = text[max_index + 1:]
        n -= 1
    else:
        digits.append(max(text))
    return int("".join(digits))



SAMPLE_INPUTS = [
    """\
987654321111111
811111111111119
234234234234278
818181911112111
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
    assert solution.solve_part_one(sample_input) == 357


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 3121910778619
