"""Advent of Code 2015, day 24: https://adventofcode.com/2015/day/24"""
from __future__ import annotations

import logging
import operator
from functools import reduce
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(24, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            weights = read_weights(f)
        first_group, *_ = find_optimal_groups(weights)
        return calc_quantum_entanglement(first_group)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            weights = read_weights(f)
        first_group, *_ = find_optimal_groups(weights, group_count=4)
        return calc_quantum_entanglement(first_group)


def read_weights(f: IO) -> list[int]:
    return [int(line.strip()) for line in f]


def find_optimal_groups(weights: list[int], group_count: int = 3) -> tuple[list[int], ...]:
    min_group_size: Optional[int] = None
    optimal_qe = 0
    result: tuple[list[int], ...] = None
    for groups in find_groups(weights, group_count):
        smallest_group, *others = sorted(groups, key=len)
        if min_group_size is None or len(smallest_group) < min_group_size:
            min_group_size = len(smallest_group)
            optimal_qe = calc_quantum_entanglement(smallest_group)
            result = smallest_group, *others
            log.debug("Found solution: %s, QE=%d", smallest_group, optimal_qe)
        elif len(smallest_group) == min_group_size:
            qe = calc_quantum_entanglement(smallest_group)
            if qe < optimal_qe:
                optimal_qe = qe
                result = smallest_group, *others
                log.debug("Found solution: %s, QE=%d", smallest_group, optimal_qe)
    return result


def find_groups(weights: list[int], group_count: int) -> Iterator[tuple[list[int], ...]]:
    if group_count == 1:
        yield weights,
        return
    if group_count < 1:
        return

    weights = sorted(weights, reverse=True)
    total_weight = sum(weights)
    target_weight = total_weight // group_count

    def _build_group(ws, target, must_include_first: bool = True) -> Iterator[list[int]]:
        if target < 0:
            return
        w, *remainder = ws
        if w == target:
            yield [w]
        if w < target and remainder:
            for subweights in _build_group(remainder, target - w, must_include_first=False):
                yield [w, *subweights]
        if remainder and not must_include_first:
            yield from _build_group(remainder, target, must_include_first=False)

    for ws in _build_group(weights, target_weight):
        reduced_weights = [n for n in weights if n not in ws]
        for groups in find_groups(reduced_weights, group_count - 1):
            result = ws, *groups
            yield result
            break  # We only need one successful combination with ws.


def calc_quantum_entanglement(weights: list[int]) -> int:
    return reduce(operator.mul, weights, 1)


SAMPLE_INPUTS = """\
1
2
3
4
5
7
8
9
10
11
"""


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS) as f:
        yield f


def test_read_weights(sample_input):
    assert read_weights(sample_input) == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]


def test_find_optimal_groups(sample_input):
    weights = read_weights(sample_input)
    result, *_ = find_optimal_groups(weights)
    assert result == [11, 9]
