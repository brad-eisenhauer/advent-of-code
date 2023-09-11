"""Advent of Code 2015, day 17: https://adventofcode.com/2015/day/17"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from io import StringIO
from typing import IO, Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            containers = read_containers(f)
        return count_storage_combos(150, containers)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            containers = read_containers(f)
        container_counts = Counter(len(c) for c in generate_storage_combos(150, containers))
        min_count = min(container_counts.keys())
        return container_counts[min_count]


def read_containers(f: IO) -> tuple[int, ...]:
    return tuple(int(c) for c in f)


def count_storage_combos(quantity: int, containers: tuple[int, ...]) -> int:
    return sum(1 for _ in generate_storage_combos(quantity, containers))


def generate_storage_combos(quantity: int, containers: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if quantity == 0:
        yield ()
        return
    if not containers:
        return
    for combo in generate_storage_combos(quantity - containers[0], containers[1:]):
        yield (containers[0], *combo)
    yield from generate_storage_combos(quantity, containers[1:])


SAMPLE_INPUTS = """\
20
15
10
5
5
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS) as f:
        yield f


@pytest.fixture
def containers(sample_input):
    return tuple(int(c) for c in sample_input)


def test_count_storage_combos(containers):
    assert count_storage_combos(25, containers) == 4


def test_generate_storage_combos(containers):
    result = list(generate_storage_combos(25, containers))
    assert result == [(20, 5), (20, 5), (15, 10), (15, 5, 5)]
