"""Advent of Code 2025, day 5: https://adventofcode.com/2025/day/5"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Optional, Self

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(5, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            inventory = Inventory.read(fp)
        return inventory.count_fresh_ingredients()

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            inventory = Inventory.read(fp)
        return inventory.count_all_fresh_ingredients()


@dataclass
class Inventory:
    fresh_ranges: list[range]
    ingredient_ids: list[int]

    @classmethod
    def read(cls, reader: IO) -> Self:
        fresh_ranges: list[range] = []
        while line := reader.readline().strip():
            lo, hi = line.split("-")
            fresh_ranges.append(range(int(lo), int(hi) + 1))
        fresh_ranges = merge_ranges(fresh_ranges)
        ingredient_ids = [int(line) for line in reader]
        return cls(fresh_ranges=fresh_ranges, ingredient_ids=ingredient_ids)

    def count_fresh_ingredients(self) -> int:
        result = 0
        ingredient_iter = iter(sorted(self.ingredient_ids))
        try:
            ing = next(ingredient_iter)
            for rng in self.fresh_ranges:
                while True:
                    if ing in rng:
                        result += 1
                    if (ing := next(ingredient_iter)) > rng.stop:
                        break
        except StopIteration:
            pass
        return result

    def count_all_fresh_ingredients(self) -> int:
        return sum(len(rng) for rng in self.fresh_ranges)


def merge_ranges(ranges: list[range]) -> list[range]:
    ordered_ranges = sorted(ranges, key=lambda r: r.start, reverse=True)
    new_ranges = [ordered_ranges.pop()]
    last_range = new_ranges[0]
    while ordered_ranges:
        next_range = ordered_ranges.pop()
        if next_range.start <= last_range.stop:
            new_ranges[-1] = range(last_range.start, max(last_range.stop, next_range.stop))
        else:
            new_ranges.append(next_range)
        last_range = new_ranges[-1]
    return new_ranges


SAMPLE_INPUTS = [
    """\
3-5
10-14
16-20
12-18

1
5
8
11
17
32
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 3


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 14
