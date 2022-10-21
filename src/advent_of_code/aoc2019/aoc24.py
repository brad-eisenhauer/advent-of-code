"""Advent of Code 2019, day 24: https://adventofcode.com/2019/day/24"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from io import StringIO
from typing import TextIO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(24, 2019)

    def solve_part_one(self) -> int:
        prior_states: set[Board] = set()
        with self.open_input() as f:
            board = Board.read(f)
        while board not in prior_states:
            prior_states.add(board)
            board = board.step()
        return board.calc_biodiversity_score()


@dataclass(frozen=True)
class Board:
    spaces: tuple[bool, ...]

    @classmethod
    def read(cls, f: TextIO) -> Board:
        result = tuple(
            char == "#" for line in f.readlines() for char in line.strip()
        )
        return cls(result)

    @cached_property
    def size(self) -> int:
        return int(math.sqrt(len(self.spaces)))

    def calc_biodiversity_score(self) -> int:
        return sum(n * 2 ** i for i, n in enumerate(self.spaces))

    def step(self) -> Board:
        neighbor_sum = self._neighbor_sum()
        result = []
        for status, neighbors in zip(self.spaces, neighbor_sum):
            match status, neighbors:
                case True, 1:
                    result.append(True)
                case True, _:
                    result.append(False)
                case (False, 1) | (False, 2):
                    result.append(True)
                case _:
                    result.append(status)
        return Board(tuple(result))

    def _neighbor_sum(self) -> list[int]:
        def coords_from_index(idx):
            return idx % self.size, idx // self.size

        def index_from_coords(x, y):
            return x + y * self.size

        result = [0] * len(self.spaces)
        valid_range = range(0, self.size)
        for index, value in enumerate(self.spaces):
            if value:
                x, y = coords_from_index(index)
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    if x + dx in valid_range and y + dy in valid_range:
                        result[index_from_coords(x + dx, y + dy)] += 1
        return result


SAMPLE_INPUTS = [
    """\
....#
#..#.
#..##
..#..
#....
""",
    """\
#..#.
####.
###.#
##.##
.##..
""",
    """\
#####
....#
....#
...#.
#.###
""",
    """\
#....
####.
...##
#.##.
.##.#
""",
    """\
####.
....#
##..#
.....
##...
""",
    """\
.....
.....
.....
#....
.#...
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(5, 2129920)],
    indirect=["sample_input"]
)
def test_calc_biodiversity_score(sample_input, expected):
    board = Board.read(sample_input)
    assert board.calc_biodiversity_score() == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, [1, 0, 0, 2, 0, 1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0])
    ],
    indirect=['sample_input']
)
def test_neighbor_sum(sample_input, expected):
    board = Board.read(sample_input)
    assert board._neighbor_sum() == expected


@pytest.mark.parametrize(
    ("sample_input", "expected_index"),
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    indirect=["sample_input"]
)
def test_step(sample_input, expected_index):
    board = Board.read(sample_input)
    with StringIO(SAMPLE_INPUTS[expected_index]) as f:
        expected_board = Board.read(f)
    assert board.step() == expected_board
