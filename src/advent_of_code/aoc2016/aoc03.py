"""Advent of Code 2016, day 3: https://adventofcode.com/2016/day/3"""
from __future__ import annotations

import logging
from io import StringIO
from typing import IO, Iterator

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            return sum(1 for t in read_triangles(fp) if triangle_is_possible(t))

    def solve_part_two(self) -> int:
        result = 0
        with self.open_input() as fp:
            for t in read_triangles_by_columns(fp):
                if triangle_is_possible(t):
                    log.debug("%s is possible.", t)
                    result += 1
                else:
                    log.debug("%s is not possible.", t)
        return result


def read_triangles(fp: IO) -> Iterator[tuple[int, ...]]:
    for line in fp:
        yield tuple(int(n) for n in line.split())


def read_triangles_by_columns(fp: IO) -> Iterator[tuple[int, ...]]:
    try:
        while True:
            l1 = tuple(int(n) for n in next(fp).split())
            l2 = tuple(int(n) for n in next(fp).split())
            l3 = tuple(int(n) for n in next(fp).split())
            yield from zip(l1, l2, l3)
    except StopIteration:
        ...


def triangle_is_possible(sides: tuple[int, ...]) -> bool:
    return sum(sides) / max(sides) > 2


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
