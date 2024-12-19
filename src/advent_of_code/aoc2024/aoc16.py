"""Advent of Code 2024, day 16: https://adventofcode.com/2024/day/16"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from io import StringIO
from typing import IO, Iterator, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar

Vector: TypeAlias = complex


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            maze = Maze.read(reader)
        nav = Navigator(maze)
        return nav.find_min_cost_to_goal(State(maze.start))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            maze = Maze.read(reader)
        nav = Navigator(maze)
        final_state, acc_cost, _ = nav._find_min_cost_path(State(maze.start))
        # Backtrace using acc_cost to find possible prior states on min cost path.
        frontier: deque[State] = deque([final_state])
        visited_locs: set[Vector] = set()
        while frontier:
            state = frontier.pop()
            visited_locs.add(state.pos)
            total_cost = acc_cost[state]
            # Find previous states; equivalent to next states from reverse facing...
            for cost, prev_state in nav.generate_next_states(replace(state, facing=-state.facing)):
                if prev_state.pos != state.pos:
                    # ...except that if we move we have to reverse facing again.
                    prev_state = replace(prev_state, facing=state.facing)
                if prev_state in acc_cost and acc_cost[prev_state] + cost == total_cost:
                    frontier.appendleft(prev_state)
        return len(visited_locs)


@dataclass
class Node:
    pos: Vector
    neighbors: dict[Vector, tuple[int, Node]] = field(default_factory=dict)


@dataclass(frozen=True)
class State:
    pos: Vector
    facing: Vector = 1

    def __lt__(self, other: State) -> bool:
        return abs(self.pos) < abs(other.pos)


@dataclass
class Maze:
    nodes: dict[Vector, Node]
    start: Vector
    end: Vector

    @classmethod
    def read(cls, reader: IO) -> Maze:
        all_nodes: dict[Vector, Node] = {}
        start_node: Node | None = None
        end_node: Node | None = None

        for y, line in enumerate(reader):
            for x, char in enumerate(line):
                pos = x + y * 1j
                match char:
                    case ".":
                        all_nodes[pos] = Node(pos)
                    case "S":
                        start_node = Node(pos)
                        all_nodes[pos] = start_node
                    case "E":
                        end_node = Node(pos)
                        all_nodes[pos] = end_node

        for node in all_nodes.values():
            facing = 1
            for _ in range(4):
                neighbor_pos = node.pos + facing
                if neighbor_pos in all_nodes:
                    node.neighbors[facing] = 1, all_nodes[neighbor_pos]
                facing *= 1j

        return cls(all_nodes, start_node.pos, end_node.pos)


class Navigator(AStar[State]):
    def __init__(self, maze: Maze) -> None:
        super().__init__()
        self.maze = maze

    def is_goal_state(self, state: State) -> bool:
        return state.pos == self.maze.end

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        node = self.maze.nodes[state.pos]
        if (neighbor := node.neighbors.get(state.facing)) is not None:
            dist, next_node = neighbor
            yield dist, replace(state, pos=next_node.pos)
        yield 1000, replace(state, facing=state.facing * 1j)
        yield 1000, replace(state, facing=state.facing * -1j)


SAMPLE_INPUTS = [
    """\
###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############
""",
    """\
#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 7036), (1, 11048)], indirect=["sample_input"]
)
def test_part_one(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 45), (1, 64)], indirect=["sample_input"]
)
def test_part_two(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_two(sample_input) == expected
