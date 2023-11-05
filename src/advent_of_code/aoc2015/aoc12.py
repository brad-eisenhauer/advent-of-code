"""Advent of Code 2015, day 12: https://adventofcode.com/2015/day/12"""
from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pytest
from multimethod import multimethod

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            struct = json.load(f)
            return sum_contents(struct)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            struct = json.load(f)
            return sum_contents(struct, red=True)


@multimethod
def sum_contents(struct: int, red: bool = False) -> int:
    return struct


@multimethod
def sum_contents(struct: dict, red: bool = False) -> int:
    if red and any(v == "red" for v in struct.values()):
        return False
    return sum(sum_contents(v, red) for v in struct.values())


@multimethod
def sum_contents(struct: list, red: bool = False) -> int:
    return sum(sum_contents(item, red) for item in struct)


@multimethod
def sum_contents(struct: Any, red: bool = False) -> int:
    return 0


SAMPLE_INPUTS = [
    """\
[1,2,3]
""",
    """\
{"a":2,"b":4}
""",
    """\
[[[3]]]
""",
    """\
{"a":{"b":4},"c":-1}
""",
    """\
{"a":[-1,1]}
""",
    """\
[-1,{"a":1}]
""",
    """\
[]
""",
    """\
{}
""",
    """\
[1,{"c":"red","b":2},3]
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected", "red"),
    [
        (0, 6, False),
        (1, 6, False),
        (2, 3, False),
        (3, 3, False),
        (4, 0, False),
        (5, 0, False),
        (6, 0, False),
        (7, 0, False),
        (0, 6, True),
        (8, 6, False),
        (8, 4, True),
    ],
    indirect=["sample_input"],
)
def test_sum_contents(sample_input, expected, red):
    struct = json.load(sample_input)
    assert sum_contents(struct, red=red) == expected
