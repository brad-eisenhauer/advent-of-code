"""Advent of Code 2024, day 6: https://adventofcode.com/2024/day/6"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from io import StringIO
from itertools import repeat
from typing import IO, Iterator, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log

Vector: TypeAlias = tuple[int, int]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            guard_map, state = GuardMap.read(reader)
        spaces = set(s.position for s in guard_map.run(state))
        return len(spaces)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            guard_map, state = GuardMap.read(reader)

        start_pos = state.position
        states: set[State] = set()
        potential_obstacles: list[tuple[Vector, State]] = []
        for state in guard_map.run(state):
            # Would adding an obstacle in front of the guard result in a loop?
            new_obstacle = vector_add(state.position, state.orientation)
            # Obstacle cannot:
            # - be initial position
            # - be out of bounds
            # - already exist
            # - have already been passed through
            if new_obstacle == start_pos:
                pass
            elif not guard_map.in_bounds(new_obstacle):
                pass
            elif new_obstacle in guard_map.obstacles:
                pass
            elif any(s.position == new_obstacle for s in states):
                pass
            else:
                potential_obstacles.append((new_obstacle, state))
            states.add(state)

        executor = ProcessPoolExecutor()

        outcomes = executor.map(_is_valid_obstacle, potential_obstacles, repeat(guard_map))
        solutions = set(o for (o, _), outcome in zip(potential_obstacles, outcomes) if outcome)
        return len(solutions)


def _is_valid_obstacle(ob_and_state: tuple[Vector, State], guard_map: GuardMap) -> bool:
    obstacle, state = ob_and_state
    new_map = replace(guard_map, obstacles=guard_map.obstacles | {obstacle})
    return not new_map.will_exit_from(state)


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(l + r for l, r in zip(left, right))


@dataclass(frozen=True)
class State:
    position: Vector
    orientation: Vector


@dataclass
class GuardMap:
    obstacles: set[Vector]
    bounds: Vector

    @classmethod
    def read(cls, reader: IO) -> tuple[GuardMap, State]:
        obstacles: set[Vector] = set()
        state: State
        for i, line in enumerate(reader):
            for j, c in enumerate(line.strip()):
                match c:
                    case "#":
                        obstacles.add((i, j))
                    case "^":
                        state = State((i, j), (-1, 0))
        return cls(obstacles=obstacles, bounds=(i, j)), state

    def step(self, state: State) -> State:
        next_pos = vector_add(state.position, state.orientation)
        if next_pos in self.obstacles:
            # rotate
            x, y = state.orientation
            return replace(state, orientation=(y, -x))
        else:
            # move
            return replace(state, position=next_pos)

    def run(self, state: State) -> Iterator[State]:
        while self.in_bounds(state.position):
            yield state
            state = self.step(state)

    def in_bounds(self, position: Vector) -> bool:
        return all(0 <= n <= b for n, b in zip(position, self.bounds))

    def will_exit_from(self, state: State) -> bool:
        prior_states: list[State] = []
        for state in self.run(state):
            if prior_states and state == prior_states[len(prior_states) // 2]:
                return False
            prior_states.append(state)
        return True


SAMPLE_INPUTS = [
    """\
....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_read(sample_input: IO):
    expected_map = GuardMap(
        obstacles={(0, 4), (1, 9), (3, 2), (4, 7), (6, 1), (7, 8), (8, 0), (9, 6)},
        bounds=(9, 9),
    )
    expected_state = State(position=(6, 4), orientation=(-1, 0))
    assert GuardMap.read(sample_input) == (expected_map, expected_state)


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 41


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 6
