"""Advent of Code 2020, day 15: https://adventofcode.com/2020/day/15"""
from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            start = read_starting_numbers(f)
        return play_game(start, 2020)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            start = read_starting_numbers(f)
        return play_game(start, 30_000_000)


def read_starting_numbers(f: TextIO) -> list[int]:
    return [int(n) for n in f.readline().split(",")]


def play_game(start: list[int], steps: int) -> int:
    record = {value: turn + 1 for turn, value in enumerate(start[:-1])}
    last_number = start[-1]
    for turn in range(len(start), steps):
        next_number = turn - record[last_number] if last_number in record else 0
        record[last_number] = turn
        last_number = next_number
    return next_number


SAMPLE_INPUTS = [
    "0,3,6",
    "1,3,2",
    "2,1,3",
    "1,2,3",
    "2,3,1",
    "3,2,1",
    "3,1,2",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 436), (1, 1), (2, 10), (3, 27), (4, 78), (5, 438), (6, 1836)],
    indirect=["sample_input"],
)
def test_play_game(sample_input, expected):
    start = read_starting_numbers(sample_input)
    assert play_game(start, 2020) == expected
