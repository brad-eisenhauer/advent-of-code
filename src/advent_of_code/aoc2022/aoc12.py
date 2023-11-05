"""Advent of Code 2022, day 12: https://adventofcode.com/2022/day/12"""
from __future__ import annotations

from io import StringIO
from typing import Callable, Iterator

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import BFS


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            map = [line.strip() for line in f]
        navigator = Navigator(map, start="S", goal="E", condition=lambda c, n: n - c <= 1)
        start = navigator.find_start()
        return navigator.find_min_cost_to_goal(start)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            map = [line.strip() for line in f]
        navigator = Navigator(map, start="E", goal="a", condition=lambda c, n: n - c >= -1)
        start = navigator.find_start()
        return navigator.find_min_cost_to_goal(start)


Position = tuple[int, int]


class Navigator(BFS[Position]):
    def __init__(
        self, map: list[str], start: str, goal: str, condition: Callable[[int, int], bool]
    ):
        self._map = map
        self._start = start
        self._goal = goal
        self._condition = condition

    def find_start(self) -> Position:
        for y, row in enumerate(self._map):
            for x, char in enumerate(row):
                if char == self._start:
                    return x, y

    def is_goal_state(self, state: Position) -> bool:
        x, y = state
        return self._map[y][x] == self._goal

    def generate_next_states(self, state: Position) -> Iterator[tuple[int, Position]]:
        x, y = state
        current_height = self.calc_height(self._map[y][x])
        for delta in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (a + b for a, b in zip(state, delta))
            if nx < 0 or ny < 0:
                continue
            if ny >= len(self._map) or nx >= len(self._map[0]):
                continue
            next_height = self.calc_height(self._map[ny][nx])
            if self._condition(current_height, next_height):
                yield 1, (nx, ny)

    @staticmethod
    def calc_height(val: str) -> int:
        if val == "S":
            return ord("a")
        if val == "E":
            return ord("z")
        return ord(val)


SAMPLE_INPUTS = [
    """\
Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(
    ("start", "end", "condition", "expected"),
    [("S", "E", (lambda c, n: n - c <= 1), 31), ("E", "a", (lambda c, n: n - c >= -1), 29)],
)
def test_find_min_cost_to_goal(sample_input, start, end, condition, expected):
    map = [line.strip() for line in sample_input]
    navigator = Navigator(map, start, end, condition)
    start = navigator.find_start()
    result = navigator.find_min_cost_to_goal(start)
    assert result == expected
