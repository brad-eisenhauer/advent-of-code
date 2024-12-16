"""Advent of Code 2024, day 2: https://adventofcode.com/2024/day/2"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            return sum(1 for line in reader if Report.parse(line).is_safe())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            return sum(1 for line in reader if Report.parse(line).is_safe(dampened=True))


@dataclass
class Report:
    levels: list[int]

    @classmethod
    def parse(cls, text: str) -> Report:
        levels = [int(n) for n in text.split()]
        return cls(levels=levels)

    def is_safe(self, dampened: bool = False) -> bool:
        def _sign(n: int) -> int:
            if n > 0:
                return 1
            if n < 0:
                return -1
            return 0

        def _safe_steps(steps: list[int]) -> bool:
            result = all(1 <= abs(s) <= 3 for s in steps) and len({_sign(s) for s in steps}) == 1
            return result

        steps = [b - a for a, b in zip(self.levels, self.levels[1:])]
        if _safe_steps(steps):
            return True
        if not dampened:
            return False

        for drop_index in range(len(self.levels)):
            new_levels = self.levels[:drop_index] + self.levels[drop_index + 1 :]
            new_steps = [b - a for a, b in zip(new_levels, new_levels[1:])]
            if _safe_steps(new_steps):
                return True
        return False


SAMPLE_INPUTS = [
    """\
7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9
5 1 2 3 4
2 4 6 2 8
2 4 6 8 2
1 2 0 6 7
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
    assert solution.solve_part_one(sample_input) == 2


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 7
