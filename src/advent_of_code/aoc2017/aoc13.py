"""Advent of Code 2017, day 13: https://adventofcode.com/2017/day/13"""

from __future__ import annotations

import re
from io import StringIO
from itertools import count
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                layer_depth, layer_range = (
                    int(match.group()) for match in re.finditer(r"\d+", line)
                )
                if layer_depth % (2 * (layer_range - 1)) == 0:
                    result += layer_depth * layer_range
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            layers = [
                tuple(int(match.group()) for match in re.finditer(r"\d+", line)) for line in reader
            ]
        for delay in count(0):
            if all((d + delay) % (2 * (r - 1)) for d, r in layers):
                return delay


SAMPLE_INPUTS = [
    """\
0: 3
1: 2
4: 4
6: 4
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
    assert solution.solve_part_one(sample_input) == 24


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 10
