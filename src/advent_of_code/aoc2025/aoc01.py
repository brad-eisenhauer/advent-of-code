"""Advent of Code 2025, day 1: https://adventofcode.com/2025/day/1"""

from __future__ import annotations

from io import StringIO
from typing import IO, Iterable, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.math import sign


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            moves = read_moves(fp)
            stops = generate_stops(moves)
            return sum(1 for stop in stops if stop == 0)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        pos = 50
        zero_passes = 0
        with input_file or self.open_input() as fp:
            moves = list(read_moves(fp))
        for move in moves:
            log.debug('Pre: {"pos": %d, "move": %d}', pos, move)
            while abs(move) >= 100:
                zero_passes += 1
                move -= sign(move) * 100
            if pos == 0:
                pos = move % 100
                continue
            if move == 0:
                continue
            new_pos = pos + move
            if new_pos <= 0 or 100 <= new_pos:
                zero_passes += 1
            pos = new_pos % 100
            log.debug('Post: {"position": %d, "zero_passes": %d}', pos, zero_passes)

        return zero_passes


def read_moves(reader: IO) -> Iterator[int]:
    for line in reader:
        yield int(line.replace("L", "-").replace("R", "+"))


def generate_stops(moves: Iterable[int], start: int = 50) -> Iterator[int]:
    for move in moves:
        start = (start + move) % 100
        yield start


SAMPLE_INPUTS = [
    """\
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
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
    assert solution.solve_part_one(sample_input) == 3


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 6
