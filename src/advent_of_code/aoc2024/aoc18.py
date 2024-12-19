"""Advent of Code 2024, day 18: https://adventofcode.com/2024/day/18"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterable, Iterator, Optional

import numpy as np
import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import BFS


class AocSolution(Solution[int, str]):
    def __init__(self, **kwargs):
        super().__init__(18, 2024, **kwargs)

    def solve_part_one(
        self, input_file: Optional[IO] = None, exit=70 + 70j, byte_count=1024
    ) -> int:
        with input_file or self.open_input() as reader:
            bytes_ = []
            for _ in range(byte_count):
                line = reader.readline()
                a, b = line.split(",")
                bytes_.append(int(a) + int(b) * 1j)
        nav = Navigator(bytes_, exit)
        return nav.find_min_cost_to_goal(State(0j))

    def solve_part_two(
        self, input_file: Optional[IO] = None, exit=70 + 70j, byte_count=1024
    ) -> str:
        with input_file or self.open_input() as reader:
            bytes_ = []
            for line in reader:
                a, b = line.split(",")
                bytes_.append(int(a) + int(b) * 1j)
        nav = Navigator([], exit)
        # Binary search for first "blocking" byte.
        hi = len(bytes_)
        lo = byte_count
        while lo + 1 < hi:
            next_index = (lo + hi) // 2
            nav.obstacles = set(bytes_[: next_index + 1])
            try:
                nav.find_min_cost_path(State(0j))
                # Path found
                lo = next_index
            except ValueError:
                # No path found
                hi = next_index
        last_byte = bytes_[hi]
        return f"{np.real(last_byte):.0f},{np.imag(last_byte):.0f}"


@dataclass(frozen=True)
class State:
    pos: complex

    def __lt__(self, other: State) -> bool:
        return abs(self.pos) < abs(other.pos)


class Navigator(BFS[State]):
    def __init__(self, obstacles: Iterable[complex], exit: complex) -> None:
        self.obstacles = set(obstacles)
        self.exit = exit
        exit_x = int(np.real(exit))
        exit_y = int(np.imag(exit))
        self.boundaries: set[complex] = (
            {-1 + n * 1j for n in range(-1, exit_y + 2)}
            | {exit_x + 1 + n * 1j for n in range(-1, exit_y + 2)}
            | {n - 1j for n in range(-1, exit_x + 2)}
            | {n + (exit_y + 1) * 1j for n in range(-1, exit_x + 2)}
        )

    def is_goal_state(self, state: State) -> bool:
        return state.pos == self.exit

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        for direction in [1, -1, 1j, -1j]:
            next_pos = state.pos + direction
            if next_pos in self.obstacles or next_pos in self.boundaries:
                continue
            yield 1, State(next_pos)


SAMPLE_INPUTS = [
    """\
5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input, exit=6 + 6j, byte_count=12) == 22


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input, exit=6 + 6j, byte_count=12) == "6,1"
