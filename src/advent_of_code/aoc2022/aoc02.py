"""Advent of Code 2022, day 2: https://adventofcode.com/2022/day/2"""

from __future__ import annotations

from enum import IntEnum
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return calc_strategy_score(f)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return calc_strategy_score(f, interpret_as_result=True)


class Shape(IntEnum):
    Rock = 1
    Paper = 2
    Scissors = 3

    @classmethod
    def from_str(cls, s: str) -> Shape:
        match s:
            case "A" | "X":
                return cls.Rock
            case "B" | "Y":
                return cls.Paper
            case "C" | "Z":
                return cls.Scissors

    def beats(self, other: Shape) -> Optional[bool]:
        if other is self:
            return None
        if other is BEATS[self]:
            return True
        return False


BEATS: dict[Shape, Shape] = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
RESULTS: dict[str, Optional[bool]] = {"X": False, "Y": None, "Z": True}


def calc_score(player: Shape, opponent: Shape) -> int:
    match player.beats(opponent):
        case True:
            return player.value + 6
        case False:
            return player.value
        case None:
            return player.value + 3


def calc_move(opponent: Shape, result: Optional[bool]) -> Shape:
    for other in Shape:
        if other.beats(opponent) is result:
            return other


def calc_strategy_score(f: TextIO, interpret_as_result: bool = False) -> int:
    if interpret_as_result:

        def calc_player_move(o: Shape, p: str) -> Shape:
            return calc_move(o, RESULTS[p])

    else:

        def calc_player_move(o: Shape, p: str) -> Shape:
            return Shape.from_str(p)

    result = 0
    for line in f:
        opponent, player = line.split()
        opponent = Shape.from_str(opponent)
        result += calc_score(calc_player_move(opponent, player), opponent)
    return result


SAMPLE_INPUTS = [
    """\
A Y
B X
C Z
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(("as_result", "expected"), [(False, 15), (True, 12)])
def test_calc_strategy_score(sample_input, as_result, expected):
    assert calc_strategy_score(sample_input, as_result) == expected
