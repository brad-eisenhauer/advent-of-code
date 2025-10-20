"""Advent of Code 2016, day 18: https://adventofcode.com/2016/day/18"""

from __future__ import annotations

from io import StringIO
from itertools import islice
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(18, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, row_count: int = 40) -> int:
        with input_file or self.open_input() as fp:
            initial_row = fp.read().strip()
        rows = islice(generate_rows(initial_row), row_count)
        result = 0
        for row in rows:
            result += sum(1 for c in row if c == ".")
        return result

    def solve_part_two(self, input_file: Optional[IO] = None, row_count: int = 400_000) -> int:
        with input_file or self.open_input() as fp:
            initial_row = fp.read().strip()
        rows = islice(generate_rows(initial_row), row_count)
        result = 0
        for row in rows:
            result += sum(1 for c in row if c == ".")
        return result


def generate_rows(initial_row: str) -> Iterator[str]:
    row = initial_row
    while True:
        yield row
        padded_row = "." + row + "."
        row = "".join("." if l == r else "^" for l, r in zip(padded_row, padded_row[2:]))


SAMPLE_INPUTS = [
    """\
..^^.
""",
    """\
.^^.^.^^^^
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
    ("sample_input", "row_count", "expected"), [(0, 3, 6), (1, 10, 38)], indirect=["sample_input"]
)
def test_part_one(solution: AocSolution, sample_input: IO, row_count: int, expected: int):
    assert solution.solve_part_one(sample_input, row_count) == expected


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...
