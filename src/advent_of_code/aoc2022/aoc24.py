"""Advent of Code 2022, day 24: https://adventofcode.com/2022/day/24"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import least_common_multiple
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(24, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            valley = Valley.parse(f)
        return calc_time_to_exit(valley)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            valley = Valley.parse(f)
        return calc_time_to_exit(valley, go_back_for_snacks=True)


Vector = tuple[int, ...]

UNIT_VECTORS: dict[str, Vector] = {">": (1, 0), "<": (-1, 0), "^": (0, -1), "v": (0, 1)}


@dataclass(frozen=True)
class Blizzard:
    loc: Vector
    movement: Vector

    def advance(self, bounds: Vector) -> Blizzard:
        x, y = (a + b for a, b in zip(self.loc, self.movement))
        return Blizzard((x % bounds[0], y % bounds[1]), self.movement)


@dataclass()
class Valley:
    blizzards: set[Blizzard]
    bounds: Vector
    enter: Vector
    exit: Vector

    def __post_init__(self):
        self.has_blizzard = cache(self._has_blizzard)

    @classmethod
    def parse(cls, f: TextIO) -> Valley:
        top = f.readline()
        enter = top.index(".") - 1
        width = len(top) - 3
        blizzards = []
        for y, line in enumerate(f):
            if line[1] == "#":
                exit = line.index(".") - 1
                continue
            for x, char in enumerate(line):
                if char in "><^v":
                    blizzards.append(Blizzard((x - 1, y), UNIT_VECTORS[char]))
        height = y
        return Valley(
            blizzards=set(blizzards),
            bounds=(width, height),
            enter=(enter, -1),
            exit=(exit, height),
        )

    def _has_blizzard(self, loc: Vector, time: int) -> bool:
        for v in UNIT_VECTORS.values():
            origin = tuple((a - time * b) % c for a, b, c in zip(loc, v, self.bounds))
            if Blizzard(origin, v) in self.blizzards:
                return True
        return False


@dataclass(frozen=True, order=True)
class State:
    elf: Vector
    time: int


class Navigator(AStar[State]):
    def __init__(self, valley: Valley):
        self._valley = valley
        self._goal = valley.exit

    def is_goal_state(self, state: State) -> bool:
        return state.elf == self._goal

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        time = state.time + 1
        for v in UNIT_VECTORS.values():
            loc = tuple(a + b for a, b in zip(state.elf, v))
            x, y = loc
            if (
                0 <= x < self._valley.bounds[0]
                and 0 <= y < self._valley.bounds[1]
                and not self._valley.has_blizzard(loc, time)
            ):
                yield 1, State(loc, time)
            if loc in [self._valley.enter, self._valley.exit]:
                yield 1, State(loc, time)
        if state.elf in [self._valley.enter, self._valley.exit] or not self._valley.has_blizzard(state.elf, time):
            yield 1, State(state.elf, time)

    def heuristic(self, state: State) -> int:
        return sum(abs(a - b) for a, b in zip(state.elf, self._valley.exit))


def calc_time_to_exit(valley: Valley, go_back_for_snacks: bool = False) -> int:
    nav = Navigator(valley)
    state = State(valley.enter, 0)
    state, _, _ = nav._find_min_cost_path(state)
    if not go_back_for_snacks:
        return state.time
    nav._goal = valley.enter
    state, _, _ = nav._find_min_cost_path(state)
    nav._goal = valley.exit
    state, _, _ = nav._find_min_cost_path(state)
    return state.time


SAMPLE_INPUTS = [
    """\
#.#####
#.....#
#>....#
#.....#
#...v.#
#.....#
#####.#
""",
    """\
#.######
#>>.<^<#
#.<..<<#
#>v.><>#
#<^v^^>#
######.#
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (
            0,
            Valley(
                blizzards={Blizzard((0, 1), (1, 0)), Blizzard((3, 3), (0, 1))},
                bounds=(5, 5),
                enter=(0, -1),
                exit=(4, 5),
            ),
        ),
        (
            1,
            Valley(
                blizzards={
                    Blizzard((0, 0), (1, 0)),
                    Blizzard((1, 0), (1, 0)),
                    Blizzard((3, 0), (-1, 0)),
                    Blizzard((4, 0), (0, -1)),
                    Blizzard((5, 0), (-1, 0)),
                    Blizzard((1, 1), (-1, 0)),
                    Blizzard((4, 1), (-1, 0)),
                    Blizzard((5, 1), (-1, 0)),
                    Blizzard((0, 2), (1, 0)),
                    Blizzard((1, 2), (0, 1)),
                    Blizzard((3, 2), (1, 0)),
                    Blizzard((4, 2), (-1, 0)),
                    Blizzard((5, 2), (1, 0)),
                    Blizzard((0, 3), (-1, 0)),
                    Blizzard((1, 3), (0, -1)),
                    Blizzard((2, 3), (0, 1)),
                    Blizzard((3, 3), (0, -1)),
                    Blizzard((4, 3), (0, -1)),
                    Blizzard((5, 3), (1, 0)),
                },
                bounds=(6, 4),
                enter=(0, -1),
                exit=(5, 4),
            ),
        ),
    ],
    indirect=["sample_input"],
)
def test_valley_parse(sample_input, expected):
    assert Valley.parse(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "loc", "time", "expected"),
    [
        (0, (1, 1), 1, True),
        (0, (0, 0), 1, False),
        (0, (3, 0), 2, True),
        (0, (0, 1), 5, True),
        (1, (2, 1), 0, False),
        (1, (2, 1), 1, False),
        (1, (2, 1), 2, True),
        (1, (2, 1), 3, True),
        (1, (2, 1), 4, False),
        (1, (2, 1), 5, True),
        (1, (2, 1), 6, True),
        (1, (2, 1), 7, False),
        (1, (2, 1), 8, True),
        (1, (2, 1), 9, True),
        (1, (2, 1), 10, True),
        (1, (2, 1), 11, True),
        (1, (2, 1), 12, False),
    ],
    indirect=["sample_input"],
)
def test_valley_has_blizzard(sample_input, loc, time, expected):
    vally = Valley.parse(sample_input)
    assert vally.has_blizzard(loc, time) is expected


@pytest.mark.parametrize(
    ("blizzard", "bounds", "expected"),
    [
        (Blizzard((0, 0), (1, 0)), (4, 4), Blizzard((1, 0), (1, 0))),
        (Blizzard((0, 3), (0, 1)), (4, 4), Blizzard((0, 0), (0, 1))),
        (Blizzard((3, 0), (1, 0)), (4, 4), Blizzard((0, 0), (1, 0))),
    ],
)
def test_blizzard_advance(blizzard, bounds, expected):
    assert blizzard.advance(bounds) == expected


@pytest.mark.parametrize(
    ("sample_input", "go_back", "expected"),
    [(0, False, 10), (1, False, 18), (1, True, 54)],
    indirect=["sample_input"],
)
def test_calc_time_to_exit(sample_input, go_back, expected):
    valley = Valley.parse(sample_input)
    assert calc_time_to_exit(valley, go_back) == expected
