"""Advent of Code 2015, day 6: https://adventofcode.com/2015/day/6"""
from __future__ import annotations

import re
from enum import IntEnum
from io import StringIO
from typing import Callable, TextIO

import numpy as np
import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return count_brightness(f, OpInterpretation.Literal)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return count_brightness(f, OpInterpretation.NorseElvish)


class OpInterpretation(IntEnum):
    Literal = 1
    NorseElvish = 2


def count_brightness(instructions: TextIO, mode: OpInterpretation) -> int:
    pattern = re.compile(r"(turn on|turn off|toggle) (\d+),(\d+) through (\d+),(\d+)")
    grid = np.zeros((1000, 1000))
    for line in instructions:
        op, *coords = pattern.match(line).groups()
        x1, y1, x2, y2 = (int(n) for n in coords)

        def apply_op(f: Callable):
            grid[x1 : x2 + 1, y1 : y2 + 1] = f(grid[x1 : x2 + 1, y1 : y2 + 1])

        match mode, op:
            case OpInterpretation.Literal, "turn on":
                apply_op(lambda g: np.logical_or(g, 1))
            case OpInterpretation.Literal, "turn off":
                apply_op(lambda g: np.logical_and(g, 0))
            case OpInterpretation.Literal, "toggle":
                apply_op(lambda g: np.logical_xor(g, 1))
            case OpInterpretation.NorseElvish, "turn on":
                apply_op(lambda g: g + 1)
            case OpInterpretation.NorseElvish, "turn off":
                apply_op(lambda g: g - 1)
                apply_op(lambda g: np.maximum(g, 0))
            case OpInterpretation.NorseElvish, "toggle":
                apply_op(lambda g: g + 2)
    return int(grid.sum().sum())


SAMPLE_INPUTS = [
    """\
turn on 0,0 through 999,999
toggle 0,0 through 999,0
turn off 499,499 through 500,500
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "mode", "expected"),
    [(0, OpInterpretation.Literal, 998996), (0, OpInterpretation.NorseElvish, 1001996)],
    indirect=["sample_input"],
)
def test_count_brightness(sample_input, mode, expected):
    assert count_brightness(sample_input, mode) == expected
