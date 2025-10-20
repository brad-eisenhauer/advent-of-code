"""Advent of Code 2016, day 19: https://adventofcode.com/2016/day/19"""

from __future__ import annotations

from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(19, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            elf_count = int(fp.read().strip())
        elves = [n + 1 for n in range(elf_count)]
        return winner_winner(elves)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            elf_count = int(fp.read().strip())
        elves = [n + 1 for n in range(elf_count)]
        return winner_winner_2(elves)


def winner_winner(elves: list[int]) -> int:
    while len(elves) > 1:
        elves = elves[0::2] if len(elves) % 2 == 0 else elves[-1:] + elves[:-1:2]
    return elves[0]


def winner_winner_2(elves: list[int]) -> int:
    while len(elves) > 1:
        # elves in the second half of the circle who survive the first pass
        survivors = elves[len(elves) // 2 + 2 - len(elves) % 2 :: 3]
        at_risk_count = len(elves) - len(elves) // 2
        tail_len = at_risk_count - len(survivors)
        # elves that took others' presents (will move to the end of the cycle)
        tail = elves[:tail_len]
        # elves who have yet to participate
        safe = elves[tail_len : len(elves) // 2]
        elves = safe + survivors + tail
    return elves[0]


SAMPLE_INPUTS = [
    """\
5
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
    assert solution.solve_part_one(sample_input) == 3


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 2
