"""Advent of Code 2019, day 24: https://adventofcode.com/2019/day/24"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from io import StringIO
from typing import ClassVar, Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(24, 2019, **kwargs)

    def solve_part_one(self) -> int:
        prior_states: set[Board] = set()
        with self.open_input() as f:
            board = Board.read(f)
        while board not in prior_states:
            prior_states.add(board)
            board = board.step()
        return board.calc_biodiversity_score()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            board = RecursiveBoard.read(f)
        for _ in range(200):
            board.step()
        return board.count_total_bugs()


@dataclass(frozen=True)
class Board:
    spaces: tuple[bool, ...]

    @classmethod
    def read(cls, f: TextIO) -> Board:
        result = tuple(char == "#" for line in f.readlines() for char in line.strip())
        return cls(result)

    @cached_property
    def size(self) -> int:
        return int(math.sqrt(len(self.spaces)))

    def calc_biodiversity_score(self) -> int:
        return sum(n * 2**i for i, n in enumerate(self.spaces))

    def step(self) -> Board:
        neighbor_sum = self._neighbor_sum()
        result = []
        for status, neighbors in zip(self.spaces, neighbor_sum):
            match status, neighbors:
                case True, 1:
                    result.append(True)
                case True, _:
                    result.append(False)
                case False, 1 | 2:
                    result.append(True)
                case _:
                    result.append(status)
        return Board(tuple(result))

    def _neighbor_sum(self) -> list[int]:
        def coords_from_index(idx):
            return idx % self.size, idx // self.size

        def index_from_coords(x, y):
            return x + y * self.size

        result = [0] * len(self.spaces)
        valid_range = range(0, self.size)
        for index, value in enumerate(self.spaces):
            if value:
                x, y = coords_from_index(index)
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    if x + dx in valid_range and y + dy in valid_range:
                        result[index_from_coords(x + dx, y + dy)] += 1
        return result


@dataclass
class RecursiveBoard:
    bugs: list[bool] = field(default_factory=lambda: [False] * 25)
    _parent: Optional[RecursiveBoard] = None
    _child: Optional[RecursiveBoard] = None
    _neighbor_counts: list[int] = field(default_factory=lambda: [0] * 25)

    CHILD_INDEX: ClassVar[int] = 12

    @classmethod
    def read(cls, f: TextIO) -> RecursiveBoard:
        result = [char == "#" for line in f.readlines() for char in line.strip()]
        return cls(result)

    @property
    def parent(self) -> RecursiveBoard:
        if self._parent is None:
            self._parent = RecursiveBoard(_child=self)
        return self._parent

    @property
    def child(self) -> RecursiveBoard:
        if self._child is None:
            self._child = RecursiveBoard(_parent=self)
        return self._child

    def step(self):
        self.clear_neighbor_counts()
        self.populate_neighbor_counts()
        self.populate_bugs()

    def count_total_bugs(self, count_all: bool = True) -> int:
        result = sum(1 for i, b in enumerate(self.bugs) if i != self.CHILD_INDEX and b)
        if not count_all:
            return result
        child = self
        while (child := child._child) is not None:
            result += child.count_total_bugs(count_all=False)
        parent = self
        while (parent := parent._parent) is not None:
            result += parent.count_total_bugs(count_all=False)
        return result

    def clear_neighbor_counts(self, clear_all: bool = True):
        self._neighbor_counts = [0] * 25
        if not clear_all:
            return
        child = self
        while (child := child._child) is not None:
            child.clear_neighbor_counts(clear_all=False)
        parent = self
        while (parent := parent._parent) is not None:
            parent.clear_neighbor_counts(clear_all=False)

    def populate_neighbor_counts(self, count_all: bool = True):
        for index, bug in enumerate(self.bugs):
            if not bug:
                continue
            x, y = self._index_to_coords(index)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor_x = x + dx
                neighbor_y = y + dy
                match neighbor_x, neighbor_y:
                    case 2, 2:  # child
                        child = self.child
                        match x, y:
                            case 1, _:  # left side
                                for child_y in range(5):
                                    child._neighbor_counts[self._coords_to_index(0, child_y)] += 1
                            case 3, _:  # right side
                                for child_y in range(5):
                                    child._neighbor_counts[self._coords_to_index(4, child_y)] += 1
                            case _, 1:  # top
                                for child_x in range(5):
                                    child._neighbor_counts[self._coords_to_index(child_x, 0)] += 1
                            case _, 3:  # bottom
                                for child_x in range(5):
                                    child._neighbor_counts[self._coords_to_index(child_x, 4)] += 1
                    case -1, _:  # parent index 11
                        self.parent._neighbor_counts[11] += 1
                    case _, -1:  # parent index 7
                        self.parent._neighbor_counts[7] += 1
                    case 5, _:  # parent index 13
                        self.parent._neighbor_counts[13] += 1
                    case _, 5:  # parent index 17
                        self.parent._neighbor_counts[17] += 1
                    case _:
                        self._neighbor_counts[self._coords_to_index(neighbor_x, neighbor_y)] += 1
        if not count_all:
            return
        child = self
        while (child := child._child) is not None:
            child.populate_neighbor_counts(count_all=False)
        parent = self
        while (parent := parent._parent) is not None:
            parent.populate_neighbor_counts(count_all=False)

    def populate_bugs(self, populate_all: bool = True):
        new_bugs = [False] * 25
        for i, (bug, neighbor_count) in enumerate(zip(self.bugs, self._neighbor_counts)):
            match bug, neighbor_count:
                case True, 1:
                    new_bugs[i] = True
                case False, 1 | 2:
                    new_bugs[i] = True
                case _:
                    ...
        self.bugs = new_bugs
        if not populate_all:
            return
        child = self
        while (child := child._child) is not None:
            child.populate_bugs(populate_all=False)
        parent = self
        while (parent := parent._parent) is not None:
            parent.populate_bugs(populate_all=False)

    @staticmethod
    def _index_to_coords(index: int) -> tuple[int, int]:
        return index % 5, index // 5

    @staticmethod
    def _coords_to_index(x: int, y: int) -> int:
        return y * 5 + x


SAMPLE_INPUTS = [
    """\
....#
#..#.
#..##
..#..
#....
""",
    """\
#..#.
####.
###.#
##.##
.##..
""",
    """\
#####
....#
....#
...#.
#.###
""",
    """\
#....
####.
...##
#.##.
.##.#
""",
    """\
####.
....#
##..#
.....
##...
""",
    """\
.....
.....
.....
#....
.#...
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(("sample_input", "expected"), [(5, 2129920)], indirect=["sample_input"])
def test_calc_biodiversity_score(sample_input, expected):
    board = Board.read(sample_input)
    assert board.calc_biodiversity_score() == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, [1, 0, 0, 2, 0, 1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0])],
    indirect=["sample_input"],
)
def test_neighbor_sum(sample_input, expected):
    board = Board.read(sample_input)
    assert board._neighbor_sum() == expected


@pytest.mark.parametrize(
    ("sample_input", "expected_index"), [(0, 1), (1, 2), (2, 3), (3, 4)], indirect=["sample_input"]
)
def test_step(sample_input, expected_index):
    board = Board.read(sample_input)
    with StringIO(SAMPLE_INPUTS[expected_index]) as f:
        expected_board = Board.read(f)
    assert board.step() == expected_board


def test_recursive():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        board = RecursiveBoard.read(f)
    for _ in range(10):
        board.step()
    assert board.count_total_bugs() == 99
