"""Advent of Code 2025, day 7: https://adventofcode.com/2025/day/7"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
from io import StringIO
from typing import IO, Optional, Self, TypeAlias

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            splitter = TachyonSplitter.read(fp)
        activated_splitters = splitter.find_activated_splitters()
        return len(activated_splitters)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            splitter = TachyonSplitter.read(fp)
        return splitter.count_possible_paths()


Vector: TypeAlias = tuple[int, int]


@dataclass(frozen=True)
class TachyonSplitter:
    origin: Vector
    splitters: frozenset[Vector]
    end: int

    @classmethod
    def read(cls, reader: IO) -> Self:
        top_line = reader.readline()
        origin = (0, top_line.index("S"))
        splitters: set[Vector] = set()
        row: int
        line: str
        for row, line in enumerate(reader, start=1):
            idx = -1
            while (idx := line.find("^", idx + 1)) >= 0:
                splitters.add((row, idx))
        return cls(origin=origin, splitters=frozenset(splitters), end=row)

    @functools.cache
    def find_activated_splitters(self, start: Vector | None = None) -> set[Vector]:
        if start is None:
            start = self.origin
        col = start[1]
        for row in range(start[0] + 1, self.end):
            if (row, col) in self.splitters:
                left = row, col - 1
                right = row, col + 1
                return {(row, col)} | self.find_activated_splitters(left) | self.find_activated_splitters(right)
        return set()

    @functools.cache
    def count_possible_paths(self, start: Vector | None = None) -> int:
        if start is None:
            start = self.origin
        col = start[1]
        for row in range(start[0] + 1, self.end):
            if (row, col) in self.splitters:
                left = row, col - 1
                right = row, col + 1
                return self.count_possible_paths(left) + self.count_possible_paths(right)
        return 1


SAMPLE_INPUTS = [
    """\
.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............
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
    assert solution.solve_part_one(sample_input) == 21


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 40
