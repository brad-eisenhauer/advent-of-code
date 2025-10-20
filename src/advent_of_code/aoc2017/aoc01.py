"""Advent of Code 2017, day 1: https://adventofcode.com/2017/day/1"""

from __future__ import annotations

from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            digit_str = fp.read().strip()
        matching_digits = (a for a, b in zip(digit_str, digit_str[1:] + digit_str[0]) if a == b)
        return sum(int(n) for n in matching_digits)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            digit_str = fp.read().strip()
        offset = len(digit_str) // 2
        matching_digits = (
            a for a, b in zip(digit_str, digit_str[offset:] + digit_str[:offset]) if a == b
        )
        return sum(int(n) for n in matching_digits)


SAMPLE_INPUTS = [
    """\
1122
""",
    """\
1111
""",
    """\
1234
""",
    """\
91212129
""",
    """\
1212
""",
    """\
1221
""",
    """\
123425
""",
    """\
123123
""",
    """\
12131415
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 3), (1, 4), (2, 0), (3, 9)], indirect=["sample_input"]
)
def test_part_one(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(4, 6), (5, 0), (6, 4), (7, 12), (8, 4)],
    indirect=["sample_input"],
)
def test_part_two(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_two(sample_input) == expected
