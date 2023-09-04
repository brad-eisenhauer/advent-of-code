"""Advent of Code 2015, day 4: https://adventofcode.com/2015/day/4"""
from __future__ import annotations

from hashlib import md5
from io import StringIO
from itertools import count

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return calc_min_number(f.readline().strip())

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return calc_min_number(f.readline().strip(), zero_prefix_len=6)


def calc_min_number(secret: str, zero_prefix_len: int = 5) -> int:
    for n in count(1):
        bytes_ = f"{secret}{n}".encode("ASCII")
        hash = md5(bytes_, usedforsecurity=False)
        if hash.hexdigest().startswith("0" * zero_prefix_len):
            return n
    raise ValueError("No number found.")


SAMPLE_INPUTS = [
    """\
abcdef
""",
    """\
pqrstuv
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 609043), (1, 1048970)], indirect=["sample_input"]
)
def test_calc_min_number(sample_input, expected):
    assert calc_min_number(sample_input.readline().strip()) == expected
