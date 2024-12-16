"""Advent of Code 2024, day 13: https://adventofcode.com/2024/day/13"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Optional

import numpy as np
import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as reader:
            for machine in read_machines(reader):
                if (cost := machine.calc_cost_to_win()) is not None:
                    result += cost
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as reader:
            for machine in read_machines(reader, correction_factor=10_000_000_000_000):
                if (cost := machine.calc_cost_to_win()) is not None:
                    result += cost
        return result


@dataclass
class Machine:
    buttons: np.ndarray
    prize: np.ndarray

    @classmethod
    def read(cls, reader: IO) -> Machine | None:
        try:
            ax, ay = (int(match.group()) for match in re.finditer(r"[+-]\d+", reader.readline()))
            bx, by = (int(match.group()) for match in re.finditer(r"[+-]\d+", reader.readline()))
            px, py = (int(match.group()) for match in re.finditer(r"\d+", reader.readline()))
            return cls(np.array([[ax, bx], [ay, by]]), np.array([px, py]))
        except:
            return None

    def calc_presses_to_win(self) -> np.ndarray | None:
        presses = np.round(np.linalg.solve(self.buttons, self.prize)).astype(int)
        if all(self.buttons @ presses == self.prize):
            return presses
        return None

    def calc_cost_to_win(self, button_costs: np.ndarray = np.array([3, 1])) -> int | None:
        if (presses := self.calc_presses_to_win()) is not None:
            return sum(button_costs * presses)
        return None


def read_machines(reader: IO, correction_factor: int = 0) -> Iterator[Machine]:
    while (machine := Machine.read(reader)) is not None:
        if correction_factor:
            machine.prize = np.array([n + correction_factor for n in machine.prize])
        yield machine
        reader.readline()


SAMPLE_INPUTS = [
    """\
Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176

Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450

Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        (Machine(np.array([[94, 22], [34, 67]]), np.array([8400, 5400])), 280),
        (Machine(np.array([[26, 67], [66, 21]]), np.array([12748, 12176])), None),
        (Machine(np.array([[17, 84], [86, 37]]), np.array([7870, 6450])), 200),
        (Machine(np.array([[69, 27], [23, 71]]), np.array([18641, 10279])), None),
    ],
)
def test_calc_cost_to_win(machine: Machine, expected: int) -> None:
    assert machine.calc_cost_to_win(np.array([3, 1])) == expected


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 480


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 875318608908
