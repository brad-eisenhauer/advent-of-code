"""Advent of Code 2022, day 24: https://adventofcode.com/2022/day/24"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass()
class Blizzard:
    loc: Vector
    movement: Vector

    def advance(self, bounds: Vector) -> Blizzard:
        x, y = (a + b for a, b in zip(self.loc, self.movement))
        return Blizzard((x % bounds[0], y % bounds[1]), self.movement)


@dataclass()
class Valley:
    blizzards: list[Blizzard]
    bounds: Vector
    enter: Vector
    exit: Vector

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
            blizzards=blizzards,
            bounds=(width, height),
            enter=(enter, -1),
            exit=(exit, height),
        )


@dataclass(frozen=True, order=True)
class State:
    elf: Vector
    time: int


class Navigator(AStar[State]):
    def __init__(self, valley: Valley):
        self._valley = valley
        self._goal = valley.exit
        self._blizzard_locs: list[set[Vector]] = [set(b.loc for b in valley.blizzards)]
        bs = valley.blizzards
        for _ in range(1, least_common_multiple(*valley.bounds)):
            bs = [b.advance(valley.bounds) for b in bs]
            self._blizzard_locs.append(set(b.loc for b in bs))

    def is_goal_state(self, state: State) -> bool:
        return state.elf == self._goal

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        next_blizzard_locs = self._blizzard_locs[(state.time + 1) % len(self._blizzard_locs)]
        for v in UNIT_VECTORS.values():
            loc = tuple(a + b for a, b in zip(state.elf, v))
            x, y = loc
            if (
                0 <= x < self._valley.bounds[0]
                and 0 <= y < self._valley.bounds[1]
                and loc not in next_blizzard_locs
            ):
                yield 1, State(loc, state.time + 1)
            if loc in [self._valley.enter, self._valley.exit]:
                yield 1, State(loc, state.time + 1)
        if state.elf not in next_blizzard_locs:
            yield 1, State(state.elf, state.time + 1)

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
                blizzards=[Blizzard((0, 1), (1, 0)), Blizzard((3, 3), (0, 1))],
                bounds=(5, 5),
                enter=(0, -1),
                exit=(4, 5),
            ),
        ),
        (
            1,
            Valley(
                blizzards=[
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
                ],
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
