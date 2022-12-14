"""Advent of Code 2022, day 9: https://adventofcode.com/2022/day/9"""
from __future__ import annotations

from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import clamp

Vector = tuple[int, ...]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            tail_positions = track_tail_movement(f)
        return len(tail_positions)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            tail_positions = track_tail_movement(f, 9)
        return len(tail_positions)


UNIT_VECTORS = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}


def track_tail_movement(f: TextIO, tail_count: int = 1) -> set[Vector]:
    knots: list[Vector] = [(0, 0)] * (tail_count + 1)
    tail_positions: set[Vector] = {knots[-1]}
    for line in f:
        d, reps = line.split()
        for _ in range(int(reps)):
            step(knots, d)
            tail_positions.add(knots[-1])
    return tail_positions


def step(knots: list[Vector], d: str):
    knots[0] = move(knots[0], UNIT_VECTORS[d])
    for i in range(1, len(knots)):
        next_knot = catch_up(knots[i - 1], knots[i])
        if next_knot == knots[i]:
            break
        knots[i] = next_knot


def move(initial: Vector, direction: Vector) -> Vector:
    return tuple(a + b for a, b in zip(initial, direction))


def catch_up(head: Vector, tail: Vector) -> Vector:
    delta = tuple(a - b for a, b in zip(head, tail))
    if max(abs(n) for n in delta) == 1:
        return tail
    movement = tuple(clamp(n, -1, 1) for n in delta)
    return move(tail, movement)


SAMPLE_INPUTS = [
    """\
R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2
""",
    """\
R 5
U 8
L 8
D 3
R 17
D 10
L 25
U 20
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "n", "expected"),
    [(0, 1, 13), (0, 9, 1), (1, 9, 36)],
    indirect=["sample_input"],
)
def test_track_tail_positions(sample_input, n, expected):
    result = track_tail_movement(sample_input, n)
    assert len(result) == expected
