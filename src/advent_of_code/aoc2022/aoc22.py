"""Advent of Code 2022, day 22: https://adventofcode.com/2022/day/22"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import IntEnum
from io import StringIO
from typing import ClassVar, Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(22, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            grid, directions = read(f)
        initial_state = find_initial_state(grid)
        for state in traverse_grid(grid, initial_state, directions):
            log.debug("%s", state)
        return calc_password(state)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            grid, directions = read(f)
        initial_state = find_initial_state(grid)
        map = {  # noqa: A001
            1: Face(
                {
                    Facing.Up: (6, Facing.Right),
                    Facing.Down: (3, Facing.Down),
                    Facing.Left: (4, Facing.Right),
                    Facing.Right: (2, Facing.Right),
                },
                (range(50, 100), range(0, 50)),
            ),
            2: Face(
                {
                    Facing.Up: (6, Facing.Up),
                    Facing.Down: (3, Facing.Left),
                    Facing.Left: (1, Facing.Left),
                    Facing.Right: (5, Facing.Left),
                },
                (range(100, 150), range(0, 50)),
            ),
            3: Face(
                {
                    Facing.Up: (1, Facing.Up),
                    Facing.Down: (5, Facing.Down),
                    Facing.Left: (4, Facing.Down),
                    Facing.Right: (2, Facing.Up),
                },
                (range(50, 100), range(50, 100)),
            ),
            4: Face(
                {
                    Facing.Up: (3, Facing.Right),
                    Facing.Down: (6, Facing.Down),
                    Facing.Left: (1, Facing.Right),
                    Facing.Right: (5, Facing.Right),
                },
                (range(0, 50), range(100, 150)),
            ),
            5: Face(
                {
                    Facing.Up: (3, Facing.Up),
                    Facing.Down: (6, Facing.Left),
                    Facing.Left: (4, Facing.Left),
                    Facing.Right: (2, Facing.Left),
                },
                (range(50, 100), range(100, 150)),
            ),
            6: Face(
                {
                    Facing.Up: (4, Facing.Up),
                    Facing.Down: (2, Facing.Down),
                    Facing.Left: (1, Facing.Down),
                    Facing.Right: (5, Facing.Up),
                },
                (range(0, 50), range(150, 200)),
            ),
        }
        for state in traverse_cube(grid, initial_state, directions, map):
            log.debug("%s", state)
        return calc_password(state)


Vector = tuple[int, ...]
Grid = list[str]


def read(f: TextIO) -> tuple[Grid, str]:
    grid = []
    while (line := f.readline().rstrip()) != "":
        grid.append(line)
    dirs = f.readline().rstrip()
    return grid, dirs


class Facing(IntEnum):
    Right = 0
    Down = 1
    Left = 2
    Up = 3

    def turn(self, direction: str) -> Facing:
        if direction == "R":
            return Facing((self + 1) % len(Facing))
        if direction == "L":
            return Facing((self - 1) % len(Facing))
        raise ValueError(f"Unrecognized direction: {direction}")


UNIT_VECTORS: dict[Facing, Vector] = {
    Facing.Right: (1, 0),
    Facing.Down: (0, 1),
    Facing.Left: (-1, 0),
    Facing.Up: (0, -1),
}


@dataclass(frozen=True)
class State:
    loc: Vector
    facing: Facing


def find_initial_state(grid: Grid) -> State:
    loc = (grid[0].index("."), 0)
    return State(loc, Facing.Right)


def traverse_grid(grid: Grid, initial_state: State, directions: str) -> Iterator[State]:
    pattern = re.compile(r"(\d+)([LR])?")
    state = initial_state
    while directions:
        match_ = pattern.match(directions)
        directions = directions[match_.end() :]

        # advance
        count = int(match_.groups()[0])
        for _ in range(count):
            x, y = (a + b for a, b in zip(state.loc, UNIT_VECTORS[state.facing]))
            y %= len(grid)
            match state.facing:
                case Facing.Up:
                    while x not in range(len(grid[y])) or grid[y][x] == " ":
                        y = (y - 1) % len(grid)
                case Facing.Down:
                    while x not in range(len(grid[y])) or grid[y][x] == " ":
                        y = (y + 1) % len(grid)
                case Facing.Left:
                    x %= len(grid[y])
                    while grid[y][x] == " ":
                        x = (x - 1) % len(grid[y])
                case Facing.Right:
                    x %= len(grid[y])
                    while grid[y][x] == " ":
                        x = (x + 1) % len(grid[y])
            if grid[y][x] == "#":
                break
            state = State((x, y), state.facing)

        # turn
        if match_.groups()[1] is not None:
            state = State(state.loc, state.facing.turn(match_.groups()[1]))

        yield state


FaceID = int


@dataclass()
class Face:
    neighbors: dict[Facing, tuple[FaceID, Facing]]
    bounds: tuple[range, ...]
    _empty_instance: ClassVar[Optional["Face"]]

    @classmethod
    def empty(cls) -> Face:
        if cls._empty_instance is None:
            cls._empty_instance = Face(0, (), {})
        return cls._empty_instance

    @property
    def is_empty(self):
        return self is self._empty_instance

    def contains(self, loc: Vector) -> bool:
        x_range, y_range = self.bounds
        x, y = loc
        return x in x_range and y in y_range


FaceMap = dict[FaceID, Face]


def traverse_cube(
    grid: Grid,
    initial_state: State,
    directions: str,
    map: FaceMap,  # noqa: A002
) -> Iterator[State]:
    pattern = re.compile(r"(\d+)([LR])?")
    state = initial_state

    def get_current_tile() -> Face:
        for tile in map.values():
            if tile.contains(state.loc):
                return tile
        raise ValueError()

    current_tile = get_current_tile()

    while directions:
        match_ = pattern.match(directions)
        directions = directions[match_.end() :]

        # advance
        count = int(match_.groups()[0])
        for _ in range(count):
            x, y = (a + b for a, b in zip(state.loc, UNIT_VECTORS[state.facing]))
            next_facing = state.facing
            if not current_tile.contains((x, y)):
                next_tile_id, next_facing = current_tile.neighbors[state.facing]
                next_tile = map[next_tile_id]
                match state.facing, next_facing:
                    case Facing.Up, Facing.Up:
                        x, y = (
                            min(next_tile.bounds[0]) + x - min(current_tile.bounds[0]),
                            max(next_tile.bounds[1]),
                        )
                    case Facing.Up, Facing.Down:
                        x, y = (
                            min(next_tile.bounds[0]) + max(current_tile.bounds[0]) - x,
                            min(next_tile.bounds[1]),
                        )
                    case Facing.Up, Facing.Left:
                        x, y = (
                            max(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + max(current_tile.bounds[0]) - x,
                        )
                    case Facing.Up, Facing.Right:
                        x, y = (
                            min(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + x - min(current_tile.bounds[0]),
                        )
                    case Facing.Down, Facing.Up:
                        x, y = (
                            min(next_tile.bounds[0]) + max(current_tile.bounds[0]) - x,
                            max(next_tile.bounds[1]),
                        )
                    case Facing.Down, Facing.Down:
                        x, y = (
                            min(next_tile.bounds[0]) + x - min(current_tile.bounds[0]),
                            min(next_tile.bounds[1]),
                        )
                    case Facing.Down, Facing.Left:
                        x, y = (
                            max(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + x - min(current_tile.bounds[0]),
                        )
                    case Facing.Down, Facing.Right:
                        x, y = (
                            min(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + max(current_tile.bounds[0]) - x,
                        )
                    case Facing.Left, Facing.Up:
                        x, y = (
                            min(next_tile.bounds[0]) + max(current_tile.bounds[1]) - y,
                            max(next_tile.bounds[1]),
                        )
                    case Facing.Left, Facing.Down:
                        x, y = (
                            min(next_tile.bounds[0]) + y - min(current_tile.bounds[1]),
                            min(next_tile.bounds[1]),
                        )
                    case Facing.Left, Facing.Left:
                        x, y = (
                            max(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + y - min(current_tile.bounds[1]),
                        )
                    case Facing.Left, Facing.Right:
                        x, y = (
                            min(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + max(current_tile.bounds[1]) - y,
                        )
                    case Facing.Right, Facing.Up:
                        x, y = (
                            min(next_tile.bounds[0]) + y - min(current_tile.bounds[1]),
                            max(next_tile.bounds[1]),
                        )
                    case Facing.Right, Facing.Down:
                        x, y = (
                            min(next_tile.bounds[0]) + max(current_tile.bounds[1]) - y,
                            min(next_tile.bounds[1]),
                        )
                    case Facing.Right, Facing.Left:
                        x, y = (
                            max(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + max(current_tile.bounds[1]) - y,
                        )
                    case Facing.Right, Facing.Right:
                        x, y = (
                            min(next_tile.bounds[0]),
                            min(next_tile.bounds[1]) + y - min(current_tile.bounds[1]),
                        )

            if grid[y][x] == "#":
                break
            state = State((x, y), next_facing)
            current_tile = get_current_tile()

        # turn
        if match_.groups()[1] is not None:
            state = State(state.loc, state.facing.turn(match_.groups()[1]))

        yield state


def calc_password(state: State) -> int:
    x, y = state.loc
    return 1000 * (y + 1) + 4 * (x + 1) + int(state.facing)


SAMPLE_INPUTS = [
    """\
        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5
""",
]

SAMPLE_FACE_MAP: FaceMap = {
    1: Face(
        {
            Facing.Up: (2, Facing.Down),
            Facing.Down: (4, Facing.Down),
            Facing.Left: (3, Facing.Down),
            Facing.Right: (6, Facing.Left),
        },
        (range(8, 12), range(0, 4)),
    ),
    2: Face(
        {
            Facing.Up: (1, Facing.Down),
            Facing.Down: (5, Facing.Up),
            Facing.Left: (6, Facing.Up),
            Facing.Right: (3, Facing.Right),
        },
        (range(0, 4), range(4, 8)),
    ),
    3: Face(
        {
            Facing.Up: (1, Facing.Right),
            Facing.Down: (5, Facing.Right),
            Facing.Left: (2, Facing.Left),
            Facing.Right: (4, Facing.Right),
        },
        (range(4, 8), range(4, 8)),
    ),
    4: Face(
        {
            Facing.Up: (1, Facing.Up),
            Facing.Down: (5, Facing.Down),
            Facing.Left: (3, Facing.Left),
            Facing.Right: (6, Facing.Down),
        },
        (range(8, 12), range(4, 8)),
    ),
    5: Face(
        {
            Facing.Up: (4, Facing.Up),
            Facing.Down: (2, Facing.Up),
            Facing.Left: (3, Facing.Up),
            Facing.Right: (6, Facing.Right),
        },
        (range(8, 12), range(8, 12)),
    ),
    6: Face(
        {
            Facing.Up: (4, Facing.Left),
            Facing.Down: (2, Facing.Right),
            Facing.Left: (5, Facing.Left),
            Facing.Right: (1, Facing.Left),
        },
        (range(12, 16), range(8, 12)),
    ),
}


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read(sample_input):
    grid, dirs = read(sample_input)
    assert grid == [
        "        ...#",
        "        .#..",
        "        #...",
        "        ....",
        "...#.......#",
        "........#...",
        "..#....#....",
        "..........#.",
        "        ...#....",
        "        .....#..",
        "        .#......",
        "        ......#.",
    ]
    assert dirs == "10R5L5R10L4R5L5"


def test_find_initial_state(sample_input):
    grid, _ = read(sample_input)
    assert find_initial_state(grid) == State((8, 0), Facing.Right)


def test_traverse_grid(sample_input):
    grid, dirs = read(sample_input)
    initial_state = find_initial_state(grid)
    result = list(traverse_grid(grid, initial_state, dirs))
    assert result == [
        State((10, 0), Facing.Down),
        State((10, 5), Facing.Right),
        State((3, 5), Facing.Down),
        State((3, 7), Facing.Right),
        State((7, 7), Facing.Down),
        State((7, 5), Facing.Right),
        State((7, 5), Facing.Right),
    ]


def test_calc_password(sample_input):
    grid, dirs = read(sample_input)
    initial_state = find_initial_state(grid)
    for state in traverse_grid(grid, initial_state, dirs):  # noqa: B007
        ...
    assert calc_password(state) == 6032


def test_traverse_cube(sample_input):
    grid, dirs = read(sample_input)
    initial_state = find_initial_state(grid)
    result = list(traverse_cube(grid, initial_state, dirs, SAMPLE_FACE_MAP))
    assert result == [
        State((10, 0), Facing.Down),
        State((10, 5), Facing.Right),
        State((14, 10), Facing.Left),
        State((10, 10), Facing.Down),
        State((1, 5), Facing.Right),
        State((6, 5), Facing.Up),
        State((6, 4), Facing.Up),
    ]
