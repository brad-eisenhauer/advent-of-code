"""Advent of Code 2017, day 9: https://adventofcode.com/2017/day/9"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import cached_property
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            groupset, _ = read_groupset(reader)
        return sum(g.score for g in groupset)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            _, result = read_groupset(reader)
        return result


@dataclass(frozen=True)
class Group:
    parent: Group | None = field(hash=False, default=None)
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    @cached_property
    def score(self) -> int:
        if self.parent is None:
            return 1
        return self.parent.score + 1


def read_groupset(stream: IO) -> tuple[set[Group], int]:
    result: set[Group] = set()
    total_garbage = 0

    def _read_garbage() -> None:
        nonlocal total_garbage
        while True:
            match stream.read(1):
                case ">":
                    return
                case "!":
                    stream.read(1)
                case _:
                    total_garbage += 1

    def _read_group(parent: Group) -> None:
        this = Group(parent)
        result.add(this)
        while True:
            match stream.read(1):
                case "<":
                    _read_garbage()
                case "{":
                    _read_group(this)
                case "}":
                    return

    while True:
        match stream.read(1):
            case "<":
                _read_garbage()
            case "{":
                _read_group(None)
            case ",":
                pass
            case c:
                return result, total_garbage


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
