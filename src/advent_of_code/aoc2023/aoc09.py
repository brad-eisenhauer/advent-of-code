"""Advent of Code 2023, day 9: https://adventofcode.com/2023/day/9"""

from __future__ import annotations

from io import StringIO
from typing import IO, Optional, Sequence

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as fp:
            for line in fp:
                ns = [int(n) for n in line.split()]
                result += extrapolate_sequence(ns)
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as fp:
            for line in fp:
                ns = [int(n) for n in line.split()][::-1]
                result += extrapolate_sequence(ns)
        return result


def extrapolate_sequence(ns: Sequence[int]) -> int:
    if not any(ns):
        return 0
    diffs = [b - a for a, b in zip(ns, ns[1:])]
    extrapolated_diff = extrapolate_sequence(diffs)
    return ns[-1] + extrapolated_diff


SAMPLE_INPUTS = [
    """\
0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 114


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 2


@pytest.mark.parametrize(("reverse", "expected"), [(False, [18, 28, 68]), (True, [-3, 0, 5])])
def test_extrapolate_sequence(sample_input, reverse, expected):
    results = []
    for line in sample_input:
        seq = [int(n) for n in line.split()]
        if reverse:
            seq = seq[::-1]
        results.append(extrapolate_sequence(seq))
    assert results == expected
