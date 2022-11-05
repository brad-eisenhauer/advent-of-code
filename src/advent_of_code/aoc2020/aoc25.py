"""Advent of Code 2020, day 25: https://adventofcode.com/2020/day/25"""
from __future__ import annotations

from io import StringIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util import mod_power


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(25, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            pk1, pk2 = map(int, (line.strip() for line in f))
        return crack_encryption_key(pk1, pk2)


def make_key(loop_size: int, subject: int = 7) -> int:
    return mod_power(subject, loop_size, 20201227)


def find_loop_size(public_key: int) -> int:
    result = 1
    candidate_key = 1
    while True:
        candidate_key *= 7
        candidate_key %= 20201227
        if candidate_key == public_key:
            return result
        result += 1


def crack_encryption_key(*pks) -> int:
    ls1, ls2 = (find_loop_size(pk) for pk in pks)
    return make_key(ls1 * ls2)


SAMPLE_INPUTS = [
    """\
5764801
17807724
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_find_loop_size(sample_input):
    pk1, pk2 = map(int, (line.strip() for line in sample_input))
    assert find_loop_size(pk1) == 8
    assert find_loop_size(pk2) == 11


def test_crack_encryption_key(sample_input):
    pk1, pk2 = map(int, (line.strip() for line in sample_input))
    result = crack_encryption_key(pk1, pk2)
    assert result == 14897079
