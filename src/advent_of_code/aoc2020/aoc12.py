"""Advent of Code 2020, day 12: https://adventofcode.com/2020/day/12"""
from dataclasses import dataclass
from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution

Vector = tuple[int, ...]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2020, **kwargs)

    def solve_part_one(self) -> int:
        ship = Ship()
        with self.open_input() as f:
            ship.navigate(f)
        return ship.calc_manhattan_dist()

    def solve_part_two(self) -> int:
        ship = Ship()
        with self.open_input() as f:
            ship.navigate_by_waypoint(f)
        return ship.calc_manhattan_dist()


@dataclass
class Ship:
    position: Vector = (0, 0)
    heading: Vector = (1, 0)
    waypoint: Vector = (10, 1)

    def navigate(self, instructions: TextIO):
        for line in instructions:
            line = line.strip()
            command, value = line[:1], int(line[1:])
            x, y = self.position
            hx, hy = self.heading
            match command, value:
                case "N", _:
                    self.position = x, y + value
                case "S", _:
                    self.position = x, y - value
                case "E", _:
                    self.position = x + value, y
                case "W", _:
                    self.position = x - value, y
                case ("L", 90) | ("R", 270):
                    self.heading = -hy, hx
                case ("R", 90) | ("L", 270):
                    self.heading = hy, -hx
                case "L" | "R", 180:
                    self.heading = -hx, -hy
                case "F", _:
                    self.position = x + value * hx, y + value * hy
                case _:
                    raise ValueError(f"Unrecognized command: {line}")

    def navigate_by_waypoint(self, instructions: TextIO):
        for line in instructions:
            line = line.strip()
            command, value = line[:1], int(line[1:])
            wx, wy = self.waypoint
            match command, value:
                case "N", _:
                    self.waypoint = wx, wy + value
                case "S", _:
                    self.waypoint = wx, wy - value
                case "E", _:
                    self.waypoint = wx + value, wy
                case "W", _:
                    self.waypoint = wx - value, wy
                case ("L", 90) | ("R", 270):
                    self.waypoint = -wy, wx
                case ("R", 90) | ("L", 270):
                    self.waypoint = wy, -wx
                case "L" | "R", 180:
                    self.waypoint = -wx, -wy
                case "F", _:
                    self.position = tuple(
                        p + value * w for p, w in zip(self.position, self.waypoint)
                    )
                case _:
                    raise ValueError(f"Unrecognized command: {line}")

    def calc_manhattan_dist(self) -> int:
        return sum(abs(x) for x in self.position)


SAMPLE_INPUTS = [
    """\
F10
N3
F7
R90
F11
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_navigate(sample_input):
    ship = Ship()
    ship.navigate(sample_input)
    assert ship.calc_manhattan_dist() == 25


def test_navigate_by_waypoint(sample_input):
    ship = Ship()
    ship.navigate_by_waypoint(sample_input)
    assert ship.calc_manhattan_dist() == 286
