"""Advent of Code 2024, day 10: https://adventofcode.com/2024/day/10"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import StringIO
from typing import IO, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.grid import pad_str_grid


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            mapper = TrailMapper.read(reader)
        return sum(len(mapper.find_reachable_summits(th)) for th in mapper.trailheads())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            mapper = TrailMapper.read(reader)
        return sum(mapper.count_distinct_paths(th) for th in mapper.trailheads())


Vector: TypeAlias = tuple[int, int]


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


DIRECTIONS: list[Vector] = [(1, 0), (0, 1), (-1, 0), (0, -1)]


@dataclass
class TrailMapper:
    topo_map: list[str]

    @classmethod
    def read(cls, reader: IO) -> TrailMapper:
        topo_map = [line.strip() for line in reader]
        topo_map = pad_str_grid(topo_map, ".", 1)
        return cls(topo_map)

    def trailheads(self) -> list[Vector]:
        return [
            (i, j)
            for i, line in enumerate(self.topo_map)
            for j, char in enumerate(line)
            if char == "0"
        ]

    def get_elevation(self, loc: Vector) -> int:
        i, j = loc
        try:
            return int(self.topo_map[i][j])
        except ValueError:
            return 99

    def find_reachable_summits(self, trailhead: Vector) -> set[Vector]:
        summits: set[Vector] = set()
        visited: set[Vector] = set()
        frontier = deque([trailhead])
        while frontier:
            loc = frontier.popleft()
            elevation = self.get_elevation(loc)
            for dir in DIRECTIONS:
                next_loc = vector_add(loc, dir)
                next_elevation = self.get_elevation(next_loc)
                if next_elevation != elevation + 1:
                    continue
                if next_elevation == 9:
                    summits.add(next_loc)
                elif next_loc not in visited:
                    frontier.append(next_loc)
                    visited.add(next_loc)
        return summits

    def count_distinct_paths(self, trailhead: Vector) -> int:
        result = 0
        frontier = deque([trailhead])
        while frontier:
            loc = frontier.popleft()
            elevation = self.get_elevation(loc)
            for dir in DIRECTIONS:
                next_loc = vector_add(loc, dir)
                next_elevation = self.get_elevation(next_loc)
                if next_elevation != elevation + 1:
                    continue
                if next_elevation == 9:
                    result += 1
                else:
                    frontier.append(next_loc)
        return result


SAMPLE_INPUTS = [
    """\
89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


@pytest.fixture()
def trail_mapper(sample_input: IO) -> TrailMapper:
    return TrailMapper.read(sample_input)


class TestTrailMapper:
    def test_read(self, trail_mapper: TrailMapper) -> None:
        assert trail_mapper == TrailMapper(
            topo_map=[
                "..........",
                ".89010123.",
                ".78121874.",
                ".87430965.",
                ".96549874.",
                ".45678903.",
                ".32019012.",
                ".01329801.",
                ".10456732.",
                "..........",
            ]
        )

    def test_trailheads(self, trail_mapper: TrailMapper) -> None:
        assert trail_mapper.trailheads() == [
            (1, 3),
            (1, 5),
            (3, 5),
            (5, 7),
            (6, 3),
            (6, 6),
            (7, 1),
            (7, 7),
            (8, 2),
        ]

    @pytest.mark.parametrize(
        ("trailhead", "expected"),
        [
            ((1, 3), {(1, 2), (4, 5), (5, 6), (6, 5), (4, 1)}),
            ((1, 5), {(1, 2), (4, 5), (5, 6), (6, 5), (4, 1), (3, 6)}),
            ((3, 5), {(1, 2), (4, 5), (5, 6), (6, 5), (4, 1)}),
            ((5, 7), {(3, 6), (4, 5), (5, 6)}),
            ((6, 3), {(7, 5)}),
            ((6, 6), {(3, 6), (4, 5), (5, 6)}),
        ],
    )
    def test_count_reachable_summits(
        self, trail_mapper: TrailMapper, trailhead: Vector, expected: int
    ) -> None:
        assert trail_mapper.find_reachable_summits(trailhead) == expected


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 36


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 81
