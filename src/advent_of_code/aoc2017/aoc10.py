"""Advent of Code 2017, day 10: https://adventofcode.com/2017/day/10"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, str]):
    def __init__(self, **kwargs):
        super().__init__(10, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, seq_len: int = 256) -> int:
        with input_file or self.open_input() as reader:
            lengths = [int(n) for n in reader.read().split(",")]
        hasher = Hasher(seq=list(range(seq_len)))
        for l in lengths:
            hasher.step(l)
        return hasher.calc_hash()

    def solve_part_two(self, input_file: Optional[IO] = None, seq_len: int = 256) -> str:
        with input_file or self.open_input() as reader:
            lengths = reader.read().strip().encode("ascii")
        lengths += bytes([17, 31, 73, 47, 23])
        hasher = Hasher(seq=list(range(seq_len)))
        for _ in range(64):
            for l in lengths:
                hasher.step(l)
        return hasher.calc_dense_hash()


@dataclass
class Hasher:
    seq: list[int]
    skip_size: int = 0
    origin: int = 0

    def step(self, length: int) -> None:
        log.debug("Stepping %d.", length)
        seq_len = len(self.seq)
        if length > 1:
            self.seq = self.seq[length - 1 :: -1] + self.seq[length:]
        movement = (length + self.skip_size) % len(self.seq)
        self.seq = self.seq[movement:] + self.seq[:movement]
        assert len(self.seq) == seq_len
        self.origin -= movement
        self.origin %= len(self.seq)
        self.skip_size += 1
        log.debug("%s", self)

    def calc_hash(self) -> int:
        return self.seq[self.origin % len(self.seq)] * self.seq[(self.origin + 1) % len(self.seq)]

    def calc_dense_hash(self) -> int:
        self.seq = self.seq[self.origin :] + self.seq[: self.origin]
        self.origin = 0
        values = [reduce(operator.xor, self.seq[i : i + 16]) for i in range(0, 256, 16)]
        return "".join(f"{n:02x}" for n in values)


SAMPLE_INPUTS = [
    """\
3,4,1,5
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution(seq_len=5)


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input, seq_len=5) == 12


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...
