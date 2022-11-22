"""Advent of Code 2015, day 3: https://adventofcode.com/2015/day/3"""
from __future__ import annotations

from io import StringIO
from typing import Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            houses = set(generate_positions(f.readline().strip()))
            return len(houses)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            houses = set(generate_positions(f.readline().strip(), n=2))
            return len(houses)


UNIT_VECTORS = {">": (1, 0), "^": (0, 1), "<": (-1, 0), "v": (0, -1)}


def generate_positions(
    directions: str, initial: tuple[int, int] = (0, 0), n: int = 1
) -> Iterator[tuple[int, int]]:
    positions: list[tuple[int, int]] = [initial] * n
    yield initial
    for i, char in enumerate(directions):
        positions[i % n] = tuple(a + b for a, b in zip(positions[i % n], UNIT_VECTORS[char]))
        yield positions[i % n]


SAMPLE_INPUTS = [
    """\
>
""",
    """\
^>v<
""",
    """\
^v^v^v^v^v
""",
    """\
^v
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, [(0, 0), (1, 0)]), (1, [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])],
    indirect=["sample_input"],
)
def test_generate_positions(sample_input, expected):
    assert list(generate_positions(sample_input.readline().strip())) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(3, [(0, 0), (0, 1), (0, -1)]), (1, [(0, 0), (0, 1), (1, 0), (0, 0), (0, 0)])],
    indirect=["sample_input"],
)
def test_alternating_positions(sample_input, expected):
    assert list(generate_positions(sample_input.readline().strip(), n=2)) == expected
