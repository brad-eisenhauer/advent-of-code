"""Advent of Code 2017, day 7: https://adventofcode.com/2017/day/7"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ...

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ...


PROGRAM_REGEX = (
    r"(?P<name>[a-z]+) "  # name followed by a space
    r"\((?P<weight>\d+)\)"  # weight surrounded by parens
    r"(?:"  # start of non-capturing, optional group
        r" -> "  # literal arrow
        r"(?P<subprog0>[a-z]+)"  # first subprogram name
        r"(?:"  # start of group
            r", (?P<subprogs>[a-z]+)"  # additional subprogram, preceded by a comma
        r")*"  # may repeat zero or more times
    r")?"  # end of optional group
)


@dataclass
class Program:
    name: str
    weight: int
    subprogram_names: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> Program:
        ...


SAMPLE_INPUTS = [
    """\
pbga (66)
xhth (57)
ebii (61)
havc (66)
ktlj (57)
fwft (72) -> ktlj, cntj, xhth
qoyq (66)
padx (45) -> pbga, havc, qoyq
tknk (41) -> ugml, padx, fwft
jptl (61)
ugml (68) -> gyxo, ebii, jptl
gyxo (61)
cntj (57)
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
