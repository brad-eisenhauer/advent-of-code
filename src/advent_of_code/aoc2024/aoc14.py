"""Advent of Code 2024, day 14: https://adventofcode.com/2024/day/14"""

from __future__ import annotations

import operator
import re
import sys
from copy import copy
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from typing import IO, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.math import least_common_multiple

Vector: TypeAlias = tuple[int, int]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, bounds: Vector = (101, 103)) -> int:
        with input_file or self.open_input() as reader:
            simulation = Simulation.read(reader, bounds)
        simulation.run(100)
        return simulation.calc_safety_factor()

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            simulation = Simulation.read(reader, bounds=(101, 103))
        min_safety_factor: int | None = None
        min_index = 0
        min_positions: list[Vector] = []
        for i in range(simulation.calc_cycle_length()):
            safety_factor = simulation.calc_safety_factor()
            if min_safety_factor is None or safety_factor < min_safety_factor:
                min_safety_factor = safety_factor
                min_index = i
                min_positions = {r.position for r in simulation.robots}
            simulation.step()
        show_positions(min_positions, simulation.bounds)
        return min_index


@dataclass
class Robot:
    position: Vector
    velocity: Vector

    @classmethod
    def read(cls, text: str) -> Robot:
        px, py, vx, vy = (int(m.group()) for m in re.finditer(r"-?\d+", text))
        return Robot((px, py), (vx, vy))

    def step(self, bounds: Vector):
        self.position = tuple((p + v) % b for p, v, b in zip(self.position, self.velocity, bounds))

    def calc_cycle_length(self, bounds: Vector) -> int:
        cycles = [least_common_multiple(v, b) // v for v, b in zip(self.velocity, bounds)]
        return reduce(least_common_multiple, cycles)


@dataclass
class Simulation:
    robots: list[Robot]
    bounds: Vector

    @classmethod
    def read(cls, reader: IO, bounds: Vector) -> Simulation:
        robots = [Robot.read(line) for line in reader]
        return cls(robots, bounds)

    def step(self) -> None:
        for robot in self.robots:
            robot.step(self.bounds)

    def run(self, step_count: int) -> None:
        for _ in range(step_count):
            self.step()

    def calc_safety_factor(self) -> int:
        quadrant_counts = [0] * 4
        boundaries = [x // 2 for x in self.bounds]
        log.debug("boundaries=%s", boundaries)
        x_boundary, y_boundary = boundaries
        for robot in self.robots:
            x, y = robot.position
            if x == x_boundary or y == y_boundary:
                # Robot is on quadrant boundary
                continue
            quadrant_index = 2 * (x // (x_boundary + 1)) + y // (y_boundary + 1)
            log.debug("robot=%s in quadrant=%d", robot, quadrant_index)
            quadrant_counts[quadrant_index] += 1
        return reduce(operator.mul, quadrant_counts)

    def calc_cycle_length(self) -> int:
        return reduce(
            least_common_multiple, (robot.calc_cycle_length(self.bounds) for robot in self.robots)
        )


def show_positions(positions: set[Vector], bounds: Vector, stream: IO = sys.stdout) -> None:
    blocks = {
        (True, True): "█",
        (True, False): "▀",
        (False, True): "▄",
        (False, False): " ",
    }
    for y in range(0, bounds[1], 2):
        line = "".join(
            blocks[((x, y) in positions, (x, y + 1) in positions)] for x in range(bounds[0])
        )
        stream.write(line)
        stream.write("\n")


SAMPLE_INPUTS = [
    """\
p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input, bounds=(11, 7)) == 12


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...
