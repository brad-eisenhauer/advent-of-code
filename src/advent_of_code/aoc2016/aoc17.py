"""Advent of Code 2016, day 17: https://adventofcode.com/2016/day/17"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import md5
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[str, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> str:
        with input_file or self.open_input() as fp:
            passcode = fp.read().strip()
        pathfinder = Pathfinder(passcode)
        *_, (result, _) = pathfinder.find_min_cost_path(State())
        return result.path

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            passcode = fp.read().strip()
        successful_states: set[State] = set()
        frontier = deque([State()])
        while frontier:
            state = frontier.popleft()
            if state.loc == (3, 3):
                successful_states.add(state)
                continue
            r_hash = md5((passcode + state.path).encode(), usedforsecurity=False).hexdigest()
            available_directions = {
                d: o for c, (d, o) in zip(r_hash, DIRECTIONS.items()) if c in "bcdef"
            }
            for direction, offset in available_directions.items():
                next_loc = tuple(a + b for a, b in zip(state.loc, offset))
                if any(n < 0 or n > 3 for n in next_loc):
                    continue
                new_state = State(next_loc, state.path + direction)
                frontier.append(new_state)
        return max(len(s.path) for s in successful_states)


Vector = tuple[int, int]


@dataclass(frozen=True)
class State:
    loc: Vector = (0, 0)
    path: str = ""

    def __lt__(self, other) -> bool:
        if not isinstance(other, State):
            raise TypeError()
        return self.loc < other.loc


DIRECTIONS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


class Pathfinder(AStar[State]):
    def __init__(self, passcode: str):
        self.passcode = passcode

    def routing_hash(self, path: str) -> str:
        return md5((self.passcode + path).encode(), usedforsecurity=False).hexdigest()

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        r_hash = self.routing_hash(state.path)
        log.debug("Evaluating %s with hash '%s'.", state, r_hash[:4])
        available_directions = {
            d: o for c, (d, o) in zip(r_hash, DIRECTIONS.items()) if c in "bcdef"
        }
        log.debug("Available directions: %s", available_directions)
        for direction, offset in available_directions.items():
            next_loc = (a + b for a, b in zip(state.loc, offset))
            if any(n < 0 or n > 3 for n in next_loc):
                continue
            yield (
                1,
                State(
                    loc=tuple(a + b for a, b in zip(state.loc, offset)), path=state.path + direction
                ),
            )

    def is_goal_state(self, state: State) -> bool:
        return state.loc == (3, 3)


SAMPLE_INPUTS = [
    """\
ihgpwlah
""",
    """\
kglvqrro
""",
    """\
ulqzkmiv
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, "DDRRRD"),
        (1, "DDUDRLRRUDRD"),
        (2, "DRURDRUDDLLDLUURRDULRLDUUDDDRR"),
    ],
    indirect=["sample_input"],
)
def test_part_one(solution: AocSolution, sample_input: IO, expected: str):
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, 370),
        (1, 492),
        (2, 830),
    ],
    indirect=["sample_input"],
)
def test_part_two(solution: AocSolution, sample_input: IO, expected: int):
    assert solution.solve_part_two(sample_input) == expected
