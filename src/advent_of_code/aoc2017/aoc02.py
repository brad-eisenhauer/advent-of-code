"""Advent of Code 2017, day 2: https://adventofcode.com/2017/day/2"""

from __future__ import annotations

from io import StringIO
from itertools import combinations
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        checksum = 0
        with input_file or self.open_input() as fp:
            for line in fp:
                values = [int(n) for n in line.split()]
                checksum += max(values) - min(values)
        return checksum

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        total = 0
        with input_file or self.open_input() as fp:
            for line in fp:
                values = sorted(int(n) for n in line.split())
                for a, b in combinations(values, 2):
                    if b % a == 0:
                        total += b // a
                        break
        return total


SAMPLE_INPUTS = [
    """\
5 1 9 5
7 5 3
2 4 6 8
""",
    """\
5 9 2 8
9 4 7 3
3 8 6 5
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 18


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 9
