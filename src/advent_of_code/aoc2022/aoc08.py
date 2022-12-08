"""Advent of Code 2022, day 8: https://adventofcode.com/2022/day/8"""
from __future__ import annotations

from io import StringIO
from typing import Callable, Iterable, Iterator, Optional, TextIO, TypeVar

import pytest

from advent_of_code.base import Solution

T = TypeVar("T")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            trees = parse_trees(f)
        return count_visible_from_edge(trees)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            trees = parse_trees(f)
        score, _ = find_max_scenic_score(trees)
        return score


def parse_trees(f: TextIO) -> list[list[int]]:
    return [[int(n) for n in line.strip()] for line in f]


def count_visible_from_edge(trees: list[list[int]]) -> int:
    dim = len(trees)
    is_visible = [[False] * dim for _ in trees]

    def mark_file(n: int, orientation: int, reverse: bool = False):
        tallest = -1
        indexes = range(dim)
        if reverse:
            indexes = reversed(indexes)
        coords = [n, n]
        for i in indexes:
            coords[orientation] = i
            x, y = coords
            height = trees[y][x]
            if height > tallest:
                is_visible[y][x] = True
                tallest = height
            if height == 9:
                return

    for n in range(dim):
        mark_file(n, 0)
        mark_file(n, 0, reverse=True)
        mark_file(n, 1)
        mark_file(n, 1, reverse=True)

    return sum(1 for row in is_visible for v in row if v)


def calc_scenic_score(trees: list[list[int]], x: int, y: int) -> int:
    dim = len(trees)
    up_score = sum(
        1 for _ in takeuntil(lambda other_y: trees[other_y][x] >= trees[y][x], range(y - 1, -1, -1))
    )
    down_score = sum(
        1 for _ in takeuntil(lambda other_y: trees[other_y][x] >= trees[y][x], range(y + 1, dim))
    )
    left_score = sum(
        1 for _ in takeuntil(lambda other_x: trees[y][other_x] >= trees[y][x], range(x - 1, -1, -1))
    )
    right_score = sum(
        1 for _ in takeuntil(lambda other_x: trees[y][other_x] >= trees[y][x], range(x + 1, dim))
    )
    return up_score * down_score * left_score * right_score


def takeuntil(predicate: Callable[[T], bool], items: Iterable[T]) -> Iterator[T]:
    for item in items:
        yield item
        if predicate(item):
            break


def find_max_scenic_score(trees: list[list[int]]) -> tuple[int, tuple[int, int]]:
    dim = len(trees)
    max_score = 0
    max_x: Optional[int] = None
    max_y: Optional[int] = None

    for x in range(1, dim - 1):
        for y in range(1, dim - 1):
            score = calc_scenic_score(trees, x, y)
            if score > max_score:
                max_score = score
                max_x = x
                max_y = y

    return max_score, (max_x, max_y)


SAMPLE_INPUTS = [
    """\
30373
25512
65332
33549
35390
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_parse_trees(sample_input):
    expected = [
        [3, 0, 3, 7, 3],
        [2, 5, 5, 1, 2],
        [6, 5, 3, 3, 2],
        [3, 3, 5, 4, 9],
        [3, 5, 3, 9, 0],
    ]
    assert parse_trees(sample_input) == expected


def test_count_visible_from_edge(sample_input):
    trees = parse_trees(sample_input)
    assert count_visible_from_edge(trees) == 21


@pytest.mark.parametrize(("x", "y", "expected"), [(2, 1, 4), (2, 3, 8)])
def test_calc_scenic_score(sample_input, x, y, expected):
    trees = parse_trees(sample_input)
    assert calc_scenic_score(trees, x, y) == expected


def test_find_max_scenic_score(sample_input):
    trees = parse_trees(sample_input)
    assert find_max_scenic_score(trees) == (8, (2, 3))
