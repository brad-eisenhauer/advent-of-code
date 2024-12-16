"""Advent of Code 2024, day 8: https://adventofcode.com/2024/day/8"""

from __future__ import annotations

from collections import defaultdict
from io import StringIO
from itertools import permutations
from typing import IO, Iterator, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            antenna_map, bounds = read_map(reader)
        results = set(find_antinodes(antenna_map, bounds))
        return len(results)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            antenna_map, bounds = read_map(reader)
        results = set(find_antinodes_any_dist(antenna_map, bounds))
        return len(results)


Vector: TypeAlias = tuple[int, int]


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(l + r for l, r in zip(left, right))


def vector_diff(left: Vector, right: Vector) -> Vector:
    return tuple(l - r for l, r in zip(left, right))


def vector_mul(v: Vector, n: int) -> Vector:
    return tuple(n * x for x in v)


def in_bounds(v: Vector, bounds: Vector) -> bool:
    return all(0 <= n <= b for n, b in zip(v, bounds))


def read_map(reader: IO) -> tuple[dict[str, list[Vector]], Vector]:
    result = defaultdict(list)
    for i, line in enumerate(reader):
        for j, char in enumerate(line.strip()):
            if "a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9":
                result[char].append((i, j))
    return result, (i, j)


def find_antinodes(antenna_map: dict[str, list[Vector]], bounds: Vector) -> Iterator[Vector]:
    for antenna_set in antenna_map.values():
        for left, right in permutations(antenna_set, 2):
            d = vector_diff(right, left)
            if in_bounds(anode := vector_add(right, d), bounds):
                yield anode


def find_antinodes_any_dist(
    antenna_map: dict[str, list[Vector]], bounds: Vector
) -> Iterator[Vector]:
    for antenna_set in antenna_map.values():
        for left, right in permutations(antenna_set, 2):
            d = vector_diff(right, left)
            while in_bounds(right, bounds):
                yield right
                right = vector_add(right, d)


SAMPLE_INPUTS = [
    """\
............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_read_map(sample_input: IO) -> None:
    sample_map, bounds = read_map(sample_input)
    assert sample_map == {
        "0": [(1, 8), (2, 5), (3, 7), (4, 4)],
        "A": [(5, 6), (8, 8), (9, 9)],
    }
    assert bounds == (11, 11)


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 14


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 34
