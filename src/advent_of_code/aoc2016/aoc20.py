"""Advent of Code 2016, day 20: https://adventofcode.com/2016/day/20"""
from __future__ import annotations

from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import union_ranges


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(20, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ranges = list(read_ranges(fp))
        sorted_ranges = sorted(ranges, key=lambda r: r.start)
        result = 0
        for r in sorted_ranges:
            if result in r:
                result = r.stop
            elif r.start > result:
                return result
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ranges = list(read_ranges(fp))
        sorted_ranges = sorted(ranges, key=lambda r: r.start)
        blocked_ranges: list[range] = sorted_ranges[:1]
        for r in sorted_ranges[1:]:
            unioned_range = union_ranges(r, blocked_ranges[-1])
            if unioned_range is not None:
                blocked_ranges[-1] = unioned_range
            else:
                blocked_ranges.append(r)
        blocked_address_count = sum(r.stop - r.start for r in blocked_ranges)
        allowed_address_count = 2**32 - blocked_address_count
        return allowed_address_count


def read_ranges(fp: IO) -> Iterator[range]:
    for line in fp:
        lo_str, hi_str = line.split("-")
        yield range(int(lo_str), int(hi_str) + 1)


SAMPLE_INPUTS = [
    """\
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
    assert solution.solve_part_one(sample_input) == ...


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...
