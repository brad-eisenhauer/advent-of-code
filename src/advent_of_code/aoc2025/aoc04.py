"""Advent of Code 2025, day 4: https://adventofcode.com/2025/day/4"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, ClassVar, Optional, Self, TypeAlias

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            diagram = PaperRollDiagram.read(fp)
        return len(diagram.find_accessible_rolls())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            diagram = PaperRollDiagram.read(fp)
        initial_roll_count = last_roll_count = len(diagram.locations)
        diagram.remove_accessible_rolls()
        while len(diagram.locations) != last_roll_count:
            last_roll_count = len(diagram.locations)
            diagram.remove_accessible_rolls()
        return initial_roll_count - len(diagram.locations)


Location: TypeAlias = tuple[int, int]
NeighborCount: TypeAlias = int


@dataclass
class PaperRollDiagram:
    locations: dict[Location, NeighborCount]

    NEIGHBORS: ClassVar[set[Location]] = {
        (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
    }

    @classmethod
    def read(cls, reader: IO) -> Self:
        locs: dict[Location, NeighborCount] = {}
        for row, line in enumerate(reader):
            for col, char in enumerate(line.strip()):
                if char == "@":
                    locs[(row, col)] = 0
        for loc in locs:
            for n in cls.NEIGHBORS:
                if add_vectors(loc, n) in locs:
                    locs[loc] += 1
        return cls(locations=locs)

    def find_accessible_rolls(self) -> set[Location]:
        return {loc for loc, n_count in self.locations.items() if n_count < 4}

    def remove_accessible_rolls(self) -> None:
        accessible_rolls = self.find_accessible_rolls()
        self.locations = {
            loc: n_count for loc, n_count in self.locations.items() if loc not in accessible_rolls
        }
        for loc in accessible_rolls:
            for n in self.NEIGHBORS:
                if (n_loc := add_vectors(loc, n)) in self.locations:
                    self.locations[n_loc] -= 1


def add_vectors(v1: Location, v2: Location) -> Location:
    return tuple(a + b for a, b in zip(v1, v2))  # type: ignore


SAMPLE_INPUTS = [
    """\
..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.
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
    assert solution.solve_part_one(sample_input) == 13


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 43
