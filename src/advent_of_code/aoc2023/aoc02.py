"""Advent of Code 2023, day 2: https://adventofcode.com/2023/day/2"""

from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from typing import IO, Iterator, Sequence

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2023, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            games = (Game.read(line.strip()) for line in fp)
            possible_games = (game for game in games if game.is_possible_given(GIVEN_BAG))
            return sum(game.id for game in possible_games)

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            games = (Game.read(line.strip()) for line in fp)
            minimal_bags = (game.find_minimal_bag() for game in games)
            powers = (calc_power(bag) for bag in minimal_bags)
            return sum(powers)


Cubes = dict[str, int]
GIVEN_BAG: Cubes = {"red": 12, "green": 13, "blue": 14}


@dataclass
class Game:
    id: int
    groups: Sequence[Cubes]

    @classmethod
    def read(cls, text: str) -> Game:
        game_id, group_str = text.split(": ")
        groups = []
        for group in group_str.split("; "):
            cubes = {}
            for color_group in group.split(", "):
                qty, color = color_group.split()
                cubes[color] = int(qty)
            groups.append(cubes)
        return cls(id=int(game_id.split()[1]), groups=groups)

    def is_possible_given(self, bag: Cubes) -> bool:
        for group in self.groups:
            if any(qty > bag.get(key, 0) for key, qty in group.items()):
                return False
        return True

    def find_minimal_bag(self) -> Cubes:
        result = defaultdict(int)
        for group in self.groups:
            for color, qty in group.items():
                result[color] = max(result[color], qty)
        return result


def calc_power(bag: Cubes) -> int:
    return reduce(operator.mul, bag.values(), 1)


SAMPLE_INPUTS = [
    """\
Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green
""",
]


@pytest.fixture
def sample_input(request) -> Iterator[IO]:
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


def test_read_game(sample_input):
    line = sample_input.readline().strip()
    assert Game.read(line) == Game(
        id=1, groups=[{"blue": 3, "red": 4}, {"red": 1, "green": 2, "blue": 6}, {"green": 2}]
    )


def test_is_possible_given(sample_input):
    given_bag = {"red": 12, "green": 13, "blue": 14}
    results = [Game.read(line.strip()).is_possible_given(given_bag) for line in sample_input]
    expected = [True, True, False, False, True]
    assert results == expected


def test_minimal_power(sample_input):
    games = (Game.read(line.strip()) for line in sample_input)
    minimal_bags = (game.find_minimal_bag() for game in games)
    results = [calc_power(bag) for bag in minimal_bags]
    assert results == [48, 12, 1560, 630, 36]
