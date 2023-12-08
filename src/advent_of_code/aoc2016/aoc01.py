"""Advent of Code 2016, day 1: https://adventofcode.com/2016/day/1"""
from __future__ import annotations

from io import StringIO
from typing import IO, Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            instructions = read_instructions(f)
        *_, dest = follow_instructions(instructions)
        return sum(abs(d) for d in dest)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            instructions = read_instructions(f)
        dest = next(find_revisited_locs(instructions))
        return sum(abs(d) for d in dest)


def read_instructions(f: IO) -> list[str]:
    return f.read().strip().split(", ")


def follow_instructions(
    instructions: list[str], intermediates: bool = False
) -> Iterator[tuple[int, int]]:
    loc = 0, 0
    facing = 0, 1

    def turn_left():
        nonlocal facing
        x, y = facing
        facing = -y, x

    def turn_right():
        nonlocal facing
        x, y = facing
        facing = y, -x

    def advance(n: int):
        nonlocal loc
        loc = tuple(l + n * v for l, v in zip(loc, facing))

    for instruction in instructions:
        turn = instruction[0]
        dist = int(instruction[1:])
        match turn:
            case "L":
                turn_left()
            case "R":
                turn_right()
        if intermediates:
            for _ in range(dist):
                advance(1)
                yield loc
        else:
            advance(dist)
            yield loc


def find_revisited_locs(instructions: list[str]) -> Iterator[tuple[int, int]]:
    visited_locations = set()
    for loc in follow_instructions(instructions, intermediates=True):
        if loc in visited_locations:
            yield loc
        else:
            visited_locations.add(loc)


SAMPLE_INPUTS = [
    "R2, L3",
    "R2, R2, R2",
    "R5, L5, R5, R3",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, ["R2", "L3"]),
        (1, ["R2"] * 3),
        (2, ["R5", "L5", "R5", "R3"]),
    ],
    indirect=["sample_input"],
)
def test_read_instructions(sample_input, expected):
    assert read_instructions(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, (2, 3)),
        (1, (0, -2)),
        (2, (10, 2)),
    ],
    indirect=["sample_input"],
)
def test_follow_instructions(sample_input, expected):
    inst = read_instructions(sample_input)
    *_, result = follow_instructions(inst)
    assert result == expected


def test_find_revisited_locations():
    inst = ["R8", "R4", "R4", "R8"]
    assert list(find_revisited_locs(inst)) == [(4, 0)]
