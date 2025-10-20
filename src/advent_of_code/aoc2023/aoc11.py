"""Advent of Code 2023, day 11: https://adventofcode.com/2023/day/11"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from io import StringIO
from itertools import combinations
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution

Galaxy = tuple[int, int]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            image = Image.read(fp)
        image = image.expand()
        return image.calc_sum_of_distances()

    def solve_part_two(
        self, input_file: Optional[IO] = None, expansion_factor: int = 1000000
    ) -> int:
        with input_file or self.open_input() as fp:
            image = Image.read(fp)
        image = image.expand(expansion_factor)
        return image.calc_sum_of_distances()


def calc_manhattan_distance(left: Galaxy, right: Galaxy) -> int:
    return sum(abs(a - b) for a, b in zip(left, right))


@dataclass(frozen=True)
class Image:
    galaxies: frozenset[Galaxy]
    dims: tuple[int, int]

    @classmethod
    def read(cls, file: IO) -> Image:
        galaxies: list[Galaxy] = []
        for row, line in enumerate(file):
            for col, char in enumerate(line.strip()):
                if char == "#":
                    galaxies.append((row, col))
        dims = row + 1, col + 1
        return cls(frozenset(galaxies), dims)

    @cached_property
    def empty_rows(self) -> tuple[int]:
        all_rows = set(range(self.dims[0]))
        return all_rows - {r for r, _ in self.galaxies}

    @cached_property
    def empty_cols(self) -> tuple[int]:
        all_cols = set(range(self.dims[1]))
        return all_cols - {c for _, c in self.galaxies}

    def expand(self, expansion_factor: int = 2) -> Image:
        new_galaxies: list[Galaxy] = []
        for galaxy in self.galaxies:
            row, col = galaxy
            new_galaxy = (
                row + sum(expansion_factor - 1 for r in self.empty_rows if r < row),
                col + sum(expansion_factor - 1 for c in self.empty_cols if c < col),
            )
            new_galaxies.append(new_galaxy)
        new_dims = self.dims[0] + len(self.empty_rows), self.dims[1] + len(self.empty_cols)
        return Image(frozenset(new_galaxies), new_dims)

    def calc_sum_of_distances(self) -> int:
        result = 0
        for left, right in combinations(self.galaxies, 2):
            result += calc_manhattan_distance(left, right)
        return result


SAMPLE_INPUTS = [
    """\
...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#.....
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
    assert solution.solve_part_one(sample_input) == 374


@pytest.mark.parametrize(("expansion_factor", "expected"), [(10, 1030), (100, 8410)])
def test_part_two(solution: AocSolution, sample_input: IO, expansion_factor, expected):
    assert solution.solve_part_two(sample_input, expansion_factor) == expected


class TestImage:
    def test_read(self, sample_input):
        result = Image.read(sample_input)
        expected = Image(
            galaxies=frozenset(
                [(0, 3), (1, 7), (2, 0), (4, 6), (5, 1), (6, 9), (8, 7), (9, 0), (9, 4)]
            ),
            dims=(10, 10),
        )
        assert result == expected

    def test_empty_columns(self, sample_input):
        image = Image.read(sample_input)
        assert image.empty_cols == {2, 5, 8}

    def test_empty_rows(self, sample_input):
        image = Image.read(sample_input)
        assert image.empty_rows == {3, 7}

    def test_expand(self, sample_input):
        image = Image.read(sample_input)
        expected = Image(
            galaxies=frozenset(
                [(0, 4), (1, 9), (2, 0), (5, 8), (6, 1), (7, 12), (10, 9), (11, 0), (11, 5)]
            ),
            dims=(12, 13),
        )
        assert image.expand() == expected
