"""Advent of Code 2025, day 1: https://adventofcode.com/2025/day/1"""

from __future__ import annotations

from io import StringIO
from typing import IO, Iterable, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.math import div


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            moves = read_moves(fp)
            stops = generate_stops(moves)
            return sum(1 for stop in stops if stop == 0)

    def solve_part_two(self, input_file: IO | None = None) -> int:
        """Count the number of times the position marker crosses or lands on zero.

        The starting position is always fifty. The space in which the marker moves is a circular
        dial with positions labelled 0-99. Each time the marker moves, we want to count the number
        of times is crosses the zero position, or if it lands on zero at the end of a move.

        Parameters
        ----------
        input_file : IO | None
            File containing moves, one per line, e.g. "R15", "L87", etc.

        Returns
        -------
        int
            The total number of times the marker passes or lands on the zero position.
        """
        pos = 50
        result = 0
        with input_file or self.open_input() as fp:
            for step in read_moves(fp):
                log.debug('Pre: {"pos": %d, "step": %d}', pos, step)
                new_passes, pos = count_zero_passes_for_single_step(pos, step)
                result += new_passes
                log.debug('Post: {"position": %d, "zero_passes": %d}', pos, result)

        return result


def read_moves(reader: IO) -> Iterator[int]:
    for line in reader:
        yield int(line.replace("L", "-").replace("R", "+"))


def generate_stops(moves: Iterable[int], start: int = 50) -> Iterator[int]:
    for move in moves:
        start = (start + move) % 100
        yield start


def count_zero_passes_for_single_step(start: int, step: int) -> tuple[int, int]:
    """Count zero passes for a single step and update the position.

    Parameters
    ----------
    start : int
        Starting position of the marker.
    step : int
        Movement to execute. Negative numbers indicate leftward/counterclockwise movement.

    Returns
    -------
    int
        Number of times the marker passed or landed on zero.
    int
        Final position of the marker.
    """
    if start == 0 and step % 100 == 0:
        return max(1, step // 100), 0
    pass_count = 0
    full_wraps, remainder = div(step, 100)
    pass_count += abs(full_wraps)
    new_pos = start + remainder
    if start and (new_pos <= 0 or 100 <= new_pos):
        pass_count += 1
    return pass_count, new_pos % 100


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
    assert solution.solve_part_two(sample_input) == 6


@pytest.mark.parametrize(
    ("start", "step", "expected"),
    [
        (25, -150, (2, 75)),
        (25, 50, (0, 75)),
        (0, 0, (1, 0)),
        (1, -2, (1, 99)),
        (1, 98, (0, 99)),
        (99, 2, (1, 1)),
        (99, -98, (0, 1)),
        (25, -50, (1, 75)),
        (99, 150, (2, 49)),
        (82, -30, (0, 52)),
        (0, -5, (0, 95)),
    ],
)
def test_count_zero_passes_for_single_move(
    start: int, step: int, expected: tuple[int, int]
) -> None:
    assert count_zero_passes_for_single_step(start, step) == expected
