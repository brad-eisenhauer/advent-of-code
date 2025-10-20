"""Advent of Code 2023, day 14: https://adventofcode.com/2023/day/14"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util import CycleDetector


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            platform = Platform.read(fp)
        return platform.tilt().calc_load()

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            platform = Platform.read(fp)

        def _spin_cycle() -> Iterator[Platform]:
            nonlocal platform
            yield platform
            while True:
                for _ in range(4):
                    platform = platform.tilt().rotate()
                yield platform

        cycles = CycleDetector(_spin_cycle())
        start, length, _ = cycles.find_cycle()
        log.debug("Found cycle starting at %d (length=%d)", start, length)
        return cycles.project_item_at(1_000_000_000).calc_load()


Vector = tuple[int, ...]


@dataclass(frozen=True)
class Platform:
    grid: tuple[str, ...]
    rocks: frozenset(Vector)
    dims: tuple[int, int]

    @classmethod
    def read(cls, file: IO) -> Platform:
        grid: list[str] = []
        rocks: set[Vector] = set()
        for row, line in enumerate(file):
            grid.append(line.strip().replace("O", "."))
            for col, char in enumerate(line.strip()):
                if char == "O":
                    rocks.add((row, col))
        return cls(tuple(grid), frozenset(rocks), (row + 1, col + 1))

    def calc_load(self):
        return sum(self.dims[0] - row for row, _ in self.rocks)

    def tilt(self) -> Platform:
        new_rocks: set[Vector] = set()
        for row, col in sorted(self.rocks):
            for r in range(row - 1, -1, -1):
                if (r, col) in new_rocks or self.grid[r][col] == "#":
                    new_rocks.add((r + 1, col))
                    break
            else:
                new_rocks.add((0, col))
        return Platform(self.grid, frozenset(new_rocks), self.dims)

    def show(self) -> str:
        grid = [
            "".join(char if (r, c) not in self.rocks else "O" for c, char in enumerate(line))
            for r, line in enumerate(self.grid)
        ]
        return "\n".join(grid)

    def rotate(self) -> Platform:
        grid = tuple("".join(line[j] for line in self.grid[::-1]) for j in range(self.dims[1]))
        rocks = frozenset((col, -row + self.dims[1] - 1) for row, col in self.rocks)
        dims = tuple(reversed(self.dims))
        return Platform(grid, rocks, dims)


SAMPLE_INPUTS = [
    """\
O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#....
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
    assert solution.solve_part_one(sample_input) == 136


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 64


class TestPlatform:
    def test_read(self, sample_input):
        expected = Platform(
            grid=(
                ".....#....",
                "....#....#",
                ".....##...",
                "...#......",
                "........#.",
                "..#....#.#",
                ".....#....",
                "..........",
                "#....###..",
                "#....#....",
            ),
            # fmt: off
            rocks=frozenset(
                [
                    (0, 0),
                    (1, 0),
                    (1, 2),
                    (1, 3),
                    (3, 0),
                    (3, 1),
                    (3, 4),
                    (3, 9),
                    (4, 1),
                    (4, 7),
                    (5, 0),
                    (5, 5),
                    (6, 2),
                    (6, 6),
                    (6, 9),
                    (7, 7),
                    (9, 1),
                    (9, 2),
                ]
            ),
            # fmt: on
            dims=(10, 10),
        )
        assert Platform.read(sample_input) == expected

    def test_tilt(self, sample_input):
        expected = Platform.read(
            StringIO(
                "\n".join(
                    [
                        "OOOO.#.O..",
                        "OO..#....#",
                        "OO..O##..O",
                        "O..#.OO...",
                        "........#.",
                        "..#....#.#",
                        "..O..#.O.O",
                        "..O.......",
                        "#....###..",
                        "#....#....",
                    ]
                )
            )
        )
        result = Platform.read(sample_input).tilt()
        log.debug("%s", result.show())
        assert result == expected

    def test_rotate(self, sample_input):
        expected = Platform(
            grid=(
                "##........",
                "..........",
                "....#.....",
                "......#...",
                "........#.",
                "##.#...#.#",
                ".#.....#..",
                ".#..#.....",
                ".....#....",
                "....#...#.",
            ),
            # fmt: off
            rocks=frozenset(
                [
                    (0, 4),
                    (0, 6),
                    (0, 8),
                    (0, 9),
                    (1, 0),
                    (1, 5),
                    (1, 6),
                    (2, 0),
                    (2, 3),
                    (2, 8),
                    (3, 8),
                    (4, 6),
                    (5, 4),
                    (6, 3),
                    (7, 2),
                    (7, 5),
                    (9, 3),
                    (9, 6),
                ]
            ),
            # fmt: on
            dims=(10, 10),
        )
        assert Platform.read(sample_input).rotate() == expected
