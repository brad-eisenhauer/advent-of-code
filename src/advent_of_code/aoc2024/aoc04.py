"""Advent of Code 2024, day 4: https://adventofcode.com/2024/day/4"""
from __future__ import annotations

from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.grid import pad_str_grid


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            grid = [line.strip() for line in reader]
        return count_xmas(grid)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            grid = [line.strip() for line in reader]
        return count_x_mas(grid)


DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def count_xmas(grid: list[str]) -> int:
    padding = 3
    padded_grid = pad_str_grid(grid, ".", padding)
    result = 0
    for i in range(padding, len(padded_grid) - padding):
        for j in range(padding, len(padded_grid[0]) - padding):
            if padded_grid[i][j] != "X":
                continue
            for direction in DIRECTIONS:
                word = "".join(
                    padded_grid[i + direction[0] * x][j + direction[1] * x] for x in range(4)
                )
                if word == "XMAS":
                    result += 1
    return result


def count_x_mas(grid: list[str]) -> int:
    padding = 1
    padded_grid = pad_str_grid(grid, ".", padding)
    result = 0
    for i in range(padding, len(padded_grid) - padding):
        for j in range(padding, len(padded_grid[0]) - padding):
            if padded_grid[i][j] != "A":
                continue
            matching_directions: set[tuple[int, int]] = set()
            for direction in DIRECTIONS:
                if 0 in direction:  # diagonals only
                    continue
                word = "".join(
                    padded_grid[i + direction[0] * x][j + direction[1] * x] for x in range(-1, 2)
                )
                if word == "MAS":
                    log.debug("Found 'MAS' at (%d, %d) offset %s.", i, j, direction)
                    matching_directions.add(direction)
            if len(matching_directions) == 2:
                result += 1
    return result


SAMPLE_INPUTS = [
    """\
MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 18


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 9
