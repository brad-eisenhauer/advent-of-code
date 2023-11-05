"""Advent of Code 2020, day 24: https://adventofcode.com/2020/day/24"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution

Vector = tuple[int, ...]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(24, 2020, **kwargs)

    def solve_part_one(self) -> int:
        grid = TileGrid()
        with self.open_input() as f:
            grid.execute(f)
        return grid.count_tiles()

    def solve_part_two(self) -> int:
        grid = TileGrid()
        with self.open_input() as f:
            grid.execute(f)
        for _ in range(100):
            grid.step()
        return grid.count_tiles()


UNIT_VECTORS: dict[str, Vector] = {
    "ne": (0, 1),
    "e": (1, 0),
    "se": (1, -1),
    "sw": (0, -1),
    "w": (-1, 0),
    "nw": (-1, 1),
}


@dataclass
class TileGrid:
    tiles: dict[Vector, bool] = field(default_factory=lambda: defaultdict(bool))

    def flip(self, directions: str):
        dest = (0, 0)
        for v in self.generate_vectors(directions):
            dest = tuple(a + b for a, b in zip(dest, v))
        self.tiles[dest] = not self.tiles[dest]

    @staticmethod
    def generate_vectors(directions) -> Iterator[Vector]:
        while directions:
            if directions[0] in ("e", "w"):
                result, directions = directions[:1], directions[1:]
            else:
                result, directions = directions[:2], directions[2:]
            yield UNIT_VECTORS[result]

    def execute(self, instructions: TextIO):
        for line in instructions:
            self.flip(line.strip())

    def step(self):
        neighbor_counts = defaultdict(int)
        for loc, state in self.tiles.items():
            if not state:
                continue
            for offset in UNIT_VECTORS.values():
                neighbor = tuple(a + b for a, b in zip(loc, offset))
                neighbor_counts[neighbor] += 1
        all_locs = set(neighbor_counts) | set(self.tiles)
        for loc in all_locs:
            match self.tiles[loc], neighbor_counts[loc]:
                case True, 0 | 3 | 4 | 5 | 6:
                    self.tiles[loc] = False
                case False, 2:
                    self.tiles[loc] = True

    def count_tiles(self) -> int:
        return sum(1 for t in self.tiles.values() if t)


SAMPLE_INPUTS = [
    """\
sesenwnenenewseeswwswswwnenewsewsw
neeenesenwnwwswnenewnwwsewnenwseswesw
seswneswswsenwwnwse
nwnwneseeswswnenewneswwnewseswneseene
swweswneswnenwsewnwneneseenw
eesenwseswswnenwswnwnwsewwnwsene
sewnenenenesenwsewnenwwwse
wenwwweseeeweswwwnwwe
wsweesenenewnwwnwsenewsenwwsesesenwne
neeswseenwwswnwswswnw
nenwswwsewswnenenewsenwsenwnesesenew
enewnwewneswsewnwswenweswnenwsenwsw
sweneswneswneneenwnewenewwneswswnese
swwesenesewenwneswnwwneseswwne
enesenwswwswneneswsenwnewswseenwsese
wnwnesenesenenwwnenwsewesewsesesew
nenewswnwewswnenesenwnesewesw
eneswnwswnwsenenwnwnwwseeswneewsenese
neswnwewnwnwseenwseesewsenwsweewe
wseweeenwnesenwwwswnew
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_flip(sample_input):
    grid = TileGrid()
    grid.execute(sample_input)
    assert grid.count_tiles() == 10


def test_step(sample_input):
    grid = TileGrid()
    grid.execute(sample_input)
    for _ in range(100):
        grid.step()
    assert grid.count_tiles() == 2208
