"""Advent of Code 2024, day 21: https://adventofcode.com/2024/day/21"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from io import StringIO
from itertools import pairwise, product
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        keypads = (NUMERIC_KEYPAD,) + (DIRECTIONAL_KEYPAD,) * 2
        result = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                line = line.strip()
                value = int(line[:-1])
                commands = enter_sequence(keypads, line)
                result += value * sum(len(r) * c for r, c in commands.items())
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        keypads = (NUMERIC_KEYPAD,) + (DIRECTIONAL_KEYPAD,) * 25
        result = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                line = line.strip()
                value = int(line[:-1])
                commands = enter_sequence(keypads, line)
                result += value * sum(len(r) * c for r, c in commands.items())
        return result


@cache
def enter_sequence(keypads: tuple[Keypad], sequence: str) -> dict[str, int]:
    keypad = keypads[0]
    route_counts: dict[str, int] = defaultdict(int)
    for left, right in pairwise("A" + sequence):
        routes = keypad.navigate(left, right)
        if len(routes) > 1:
            route = min(routes, key=calc_min_successor_sequence_length)
        else:
            route = routes[0]
        route_counts[route] += 1

    if len(keypads) == 1:
        return route_counts

    result: dict[str, int] = defaultdict(int)
    for route, count in route_counts.items():
        next_routes = enter_sequence(keypads[1:], route)
        for nr, c in next_routes.items():
            result[nr] += count * c

    return result


def calc_min_successor_sequence_length(sequence: str, depth: int = 4) -> int:
    candidate_parts: list[list[str]] = [
        DIRECTIONAL_KEYPAD.navigate(left, right) for left, right in pairwise("A" + sequence)
    ]
    candidates: list[list[str]] = list(product(*candidate_parts))
    if depth == 0:
        return min(sum(len(part) for part in candidate) for candidate in candidates)
    return min(
        sum(calc_min_successor_sequence_length(part, depth - 1) for part in candidate)
        for candidate in candidates
    )


@dataclass
class Keypad:
    keys: dict[str, complex]

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        self.values = set(self.keys.values())

    def _navigate(self, origin: complex, target: complex) -> Iterator[str]:
        diff = target - origin
        corners = [origin + diff.real, origin + diff.imag * 1j]
        if diff.real > 0:
            horiz_part = ">" * int(diff.real)
        else:
            horiz_part = "<" * int(abs(diff.real))
        if diff.imag > 0:
            vert_part = "^" * int(diff.imag)
        else:
            vert_part = "v" * int(abs(diff.imag))
        seqs = [horiz_part + vert_part, vert_part + horiz_part]
        yield from set(seq for seq, corner in zip(seqs, corners) if corner in self.values)

    @cache
    def navigate(self, origin: str, target: str) -> list[str]:
        return list(seq + "A" for seq in self._navigate(self.keys[origin], self.keys[target]))


NUMERIC_KEYPAD = Keypad(
    keys={
        "0": 0,
        "A": 1,
        "1": -1 + 1j,
        "2": 1j,
        "3": 1 + 1j,
        "4": -1 + 2j,
        "5": 2j,
        "6": 1 + 2j,
        "7": -1 + 3j,
        "8": 3j,
        "9": 1 + 3j,
    }
)
DIRECTIONAL_KEYPAD = Keypad(keys={"<": -1, "v": 0, ">": 1, "^": 1j, "A": 1 + 1j})


SAMPLE_INPUTS = [
    """\
029A
980A
179A
456A
379A
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 126384
