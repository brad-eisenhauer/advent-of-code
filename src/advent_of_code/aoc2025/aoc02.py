"""Advent of Code 2025, day 2: https://adventofcode.com/2025/day/2"""

from __future__ import annotations

import re
from io import StringIO
from itertools import count
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ranges = read_ranges(fp.read())
        ranges.sort()
        max_value = ranges[-1][-1]
        invalid_sum = 0
        for invalid_id in generate_invalid_ids():
            if invalid_id > max_value:
                break
            if any(left <= invalid_id <= right for left, right in ranges):
                invalid_sum += invalid_id
        return invalid_sum

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ranges = read_ranges(fp.read())
        invalid_sum = 0
        for left, right in ranges:
            for n in range(left, right + 1):
                if not is_valid(n):
                    invalid_sum += n
        return invalid_sum


def read_ranges(text: str) -> list[tuple[int, int]]:
    range_strs = text.split(",")
    result = []
    for rng in range_strs:
        left, right, *_ = rng.split("-")
        result.append((int(left), int(right)))
    return result


def is_valid(product_id: int) -> bool:
    return re.match(r"^(\d+)\1+$", str(product_id)) is None


def generate_invalid_ids() -> Iterator[int]:
    for n in count(1):
        yield int(f"{n}{n}")


SAMPLE_INPUTS = [
    """\
11-22,95-115,998-1012,1188511880-1188511890,222220-222224,\
1698522-1698528,446443-446449,38593856-38593862,565653-565659,\
824824821-824824827,2121212118-2121212124\
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
    assert solution.solve_part_one(sample_input) == 1227775554


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 4174379265
