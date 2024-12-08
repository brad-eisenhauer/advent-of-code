"""Advent of Code 2017, day 5: https://adventofcode.com/2017/day/5"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Literal, Optional

import pytest
from typing_extensions import Self

from advent_of_code.base import PuzzlePart, Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(5, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            runner = Runner.read(fp)
        return sum(1 for _ in runner.run())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            runner = Runner.read(fp, rule=PuzzlePart.Two)
        return sum(1 for _ in runner.run())


@dataclass
class Runner:
    register: list[int]
    rule: Literal[PuzzlePart.One, PuzzlePart.Two] = PuzzlePart.One

    @classmethod
    def read(cls, reader: IO, rule: PuzzlePart = PuzzlePart.One) -> Self:
        register = [int(line.strip()) for line in reader]
        return cls(register, rule)

    def run(self) -> Iterator[int]:
        index = 0
        yield 0
        while (index := self.step(index)) in range(len(self.register)):
            yield index

    def step(self, index: int) -> int:
        next_index = index + self.register[index]
        match self.rule:
            case PuzzlePart.One:
                self.register[index] += 1
            case PuzzlePart.Two:
                self.register[index] += 1 if self.register[index] < 3 else -1
        return next_index


SAMPLE_INPUTS = [
    """\
0
3
0
1
-3
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
    assert solution.solve_part_one(sample_input) == 5


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 10
