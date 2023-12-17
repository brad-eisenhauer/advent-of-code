"""Advent of Code 2023, day 17: https://adventofcode.com/2023/day/17"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            grid = [line.strip() for line in fp]
        pathfinder = Pathfinder(grid)
        initial_state = State((0, 0))
        return pathfinder.find_min_cost_to_goal(initial_state)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            grid = [line.strip() for line in fp]
        pathfinder = Pathfinder(grid, move_range=range(4, 11))
        initial_state = State((0, 0))
        return pathfinder.find_min_cost_to_goal(initial_state)


Vector = tuple[int, int]


@dataclass(frozen=True)
class State:
    position: Vector
    orientation: Optional[Vector] = None
    moves_since_turn: int = 0

    def __lt__(self, other) -> bool:
        if not isinstance(other, State):
            raise TypeError()
        return self.position < other.position


class Pathfinder(AStar[State]):
    def __init__(self, grid: list[str], move_range: range = range(4)) -> None:
        super().__init__()
        self.grid = grid
        self.move_range = move_range

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        log.debug("From state %s:", state)
        for move in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if state.orientation == tuple(-n for n in move):
                log.debug("- movement %s would reverse. Skipping.", move)
                continue
            moves_since_turn = 1 if move != state.orientation else state.moves_since_turn + 1
            if (
                state.orientation is not None
                and state.orientation != move
                and state.moves_since_turn < min(self.move_range)
            ):
                log.debug(
                    "- movement %s would turn too soon (%d blocks). Skipping",
                    move,
                    moves_since_turn,
                )
                continue
            if moves_since_turn > max(self.move_range):
                log.debug(
                    "- movement %s would continue for %d blocks. Skipping.", move, moves_since_turn
                )
                continue
            next_pos = tuple(a + b for a, b in zip(move, state.position))
            row, col = next_pos
            if row < 0 or len(self.grid) <= row or col < 0 or len(self.grid[0]) <= col:
                log.debug("- movement %s exits the grid at %s. Skipping.", move, next_pos)
                continue
            cost = int(self.grid[row][col])
            next_state = State(next_pos, move, moves_since_turn)
            log.debug("- next state %s with cost %d.", next_state, cost)
            yield cost, next_state

    @cached_property
    def goal_state(self) -> Vector:
        return len(self.grid) - 1, len(self.grid[0]) - 1

    def is_goal_state(self, state: State) -> bool:
        return state.position == self.goal_state

    def heuristic(self, state: State) -> int:
        return sum(abs(a - b) for a, b in zip(state.position, self.goal_state))


SAMPLE_INPUTS = [
    """\
2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 102


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...
