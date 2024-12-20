"""Advent of Code 2024, day 20: https://adventofcode.com/2024/day/20"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import BFS


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(20, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, min_time_saved: int = 100) -> int:
        with input_file or self.open_input() as reader:
            maze = Maze.read(reader)
        nav = Navigator(maze)
        return nav.count_shortcuts(2, min_time_saved)

    def solve_part_two(self, input_file: Optional[IO] = None, min_time_saved: int = 100) -> int:
        with input_file or self.open_input() as reader:
            maze = Maze.read(reader)
        nav = Navigator(maze)
        return nav.count_shortcuts(20, min_time_saved)


def generate_points_within(origin: complex, max_dist: int) -> Iterator[tuple[int, complex]]:
    for x_offset in range(-max_dist, max_dist + 1):
        for y_offset in range(-max_dist + abs(x_offset), max_dist - abs(x_offset) + 1):
            dist = abs(x_offset) + abs(y_offset)
            yield dist, origin + x_offset + y_offset * 1j


@dataclass(frozen=True)
class State:
    pos: complex

    def __lt__(self, other: State) -> bool:
        return abs(self.pos) < abs(other.pos)


@dataclass
class Maze:
    start: complex
    end: complex
    walls: set[complex]

    @classmethod
    def read(cls, reader: IO) -> Maze:
        start: complex | None = None
        end: complex | None = None
        walls: set[complex] = set()
        for i, line in enumerate(reader):
            for j, char in enumerate(line):
                pos = i + j * 1j
                match char:
                    case "S":
                        start = pos
                    case "E":
                        end = pos
                    case "#":
                        walls.add(pos)
        if start is None:
            raise ValueError("Start position not found.")
        if end is None:
            raise ValueError("End position not found.")
        return Maze(start, end, walls)


@dataclass
class Navigator(BFS[State]):
    maze: Maze

    def is_goal_state(self, state: State) -> bool:
        return state.pos == self.maze.end

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        for direction in [1, -1, 1j, -1j]:
            next_pos = state.pos + direction
            if next_pos not in self.maze.walls:
                yield 1, replace(state, pos=next_pos)

    def count_shortcuts(self, max_length: int, min_time_saved: int) -> int:
        _, accumulated_cost, _ = self._find_min_cost_path(State(self.maze.start))
        result = 0
        for state, current_time in accumulated_cost.items():
            for shortcut_dist, destination in generate_points_within(state.pos, max_length):
                dest_state = State(destination)
                if dest_state not in accumulated_cost:
                    continue
                dest_time = current_time + shortcut_dist
                time_saved = accumulated_cost[dest_state] - dest_time
                if time_saved >= min_time_saved:
                    result += 1
        return result


SAMPLE_INPUTS = [
    """\
###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_maze_read(sample_input: IO) -> None:
    maze = Maze.read(sample_input)
    assert maze.start == 3 + 1j
    assert maze.end == 7 + 5j


def test_navigator(sample_input: IO) -> None:
    maze = Maze.read(sample_input)
    nav = Navigator(maze)
    initial_state = State(maze.start)
    assert nav.find_min_cost_to_goal(initial_state) == 84


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input, min_time_saved=20) == 5


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input, min_time_saved=70) == 41
