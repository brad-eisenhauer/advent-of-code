"""Advent of Code 2015, day 25: https://adventofcode.com/2015/day/25"""
from __future__ import annotations

import logging
import re
from io import StringIO
from typing import IO

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(25, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            row, col = read_input(f)
        log.debug("Calculating for row=%d, col=%d", row, col)
        cell_index = calc_cell_index(row, col)
        return calc_code(cell_index)


def read_input(f: IO) -> tuple[int, int]:
    return tuple(int(m) for m in re.findall(r"\d+", f.read()))


def calc_cell_index(row, col) -> int:
    result = calc_triangle_number(col)
    result += int(round(row**2 / 2 + (2 * col - 3) * row / 2 - col + 1))
    return result


def calc_triangle_number(n: int) -> int:
    return n * (n + 1) // 2


def calc_code(index: int) -> int:
    result = 20151125
    for _ in range(index - 1):
        result *= 252533
        result %= 33554393
    return result


SAMPLE_INPUTS = [
    """\
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(("sample_input", "expected"), [], indirect=["sample_input"])
def test_foo(sample_input, expected):
    ...


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [(1, 1, 1), (25, 1, 301), (1, 9, 45), (25, 9, 537)],
)
def test_cell_index(row, col, expected):
    assert calc_cell_index(row, col) == expected


@pytest.mark.parametrize(
    ("cell_index", "expected"), [(1, 20151125), (21, 33511524), (16, 33071741)]
)
def test_calc_code(cell_index, expected):
    assert calc_code(cell_index) == expected
