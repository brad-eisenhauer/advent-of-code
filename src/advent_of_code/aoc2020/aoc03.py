"""Advent of Code 2020, day 3: https://adventofcode.com/2020/day/3"""
from __future__ import annotations

from io import StringIO
from typing import Iterable

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            trees = TreeMap.parse(f)
        return trees.count_trees((3, 1))

    def solve_part_two(self) -> int:
        trajectories = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
        result = 1
        with self.open_input() as f:
            trees = TreeMap.parse(f)
        for traj in trajectories:
            result *= trees.count_trees(traj)
        return result


class TreeMap:
    @classmethod
    def parse(cls, lines: Iterable[str]) -> TreeMap:
        trees = set()
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == "#":
                    trees.add((x, y))
        return TreeMap(trees, x, y + 1)

    def __init__(self, trees: set[tuple[int, int]], width: int, height: int):
        self.trees = trees
        self.width = width
        self.height = height

    def is_tree(self, x: int, y: int) -> bool:
        return (x, y) in self.trees

    def count_trees(self, trajectory: tuple[int, int]) -> int:
        dx, dy = trajectory
        x = y = 0
        result = 0
        while True:
            if (x, y) in self.trees:
                result += 1
            y += dy
            if y >= self.height:
                break
            x += dx
            x %= self.width
        return result


SAMPLE_INPUT = """\
..##.......
#...#...#..
.#....#..#.
..#.#...#.#
.#...##..#.
..#.##.....
.#.#.#....#
.#........#
#.##...#...
#...##....#
.#..#...#.#
"""


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_parse_tree_map(sample_input):
    result = TreeMap.parse(sample_input)
    expected_count = sum(1 for c in SAMPLE_INPUT if c == "#")
    assert len(result.trees) == expected_count
    assert result.width == 11
    assert result.height == 11
    for coord in [(2, 0), (3, 0), (0, 1), (4, 1), (8, 1), (10, 10)]:
        assert coord in result.trees


def test_count_trees(sample_input):
    trees = TreeMap.parse(sample_input)
    result = trees.count_trees((3, 1))
    assert result == 7
