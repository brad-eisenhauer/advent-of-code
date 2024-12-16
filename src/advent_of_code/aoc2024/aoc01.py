"""Advent of Code 2024, day 1: https://adventofcode.com/2024/day/1"""

from __future__ import annotations

from collections import Counter
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            left, right = read_input(reader)
        return sum(abs(a - b) for a, b in zip(sorted(left), sorted(right)))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            left, right = read_input(reader)
        counts = Counter(right)
        return sum(x * counts[x] for x in left if x in counts)


def read_input(reader: IO) -> tuple[list[int], list[int]]:
    left = []
    right = []
    for line in reader:
        a, b = (int(x) for x in line.split())
        left.append(a)
        right.append(b)
    return left, right


SAMPLE_INPUTS = [
    """\
3   4
4   3
2   5
1   3
3   9
3   3
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
    assert solution.solve_part_one(sample_input) == 11


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 31


def test_read_input(sample_input: IO) -> None:
    left, right = read_input(sample_input)
    assert left == [3, 4, 2, 1, 3, 3]
    assert right == [4, 3, 5, 3, 9, 3]
