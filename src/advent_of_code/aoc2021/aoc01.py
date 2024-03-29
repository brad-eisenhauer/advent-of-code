"""Advent of Code 2021, Day 01: https://adventofcode.com/2021/day/1"""
from io import StringIO
from itertools import islice, tee
from typing import Iterable, Iterator, TextIO, Tuple, TypeVar

import pytest

from advent_of_code.base import Solution

T = TypeVar("T")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(1, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            measurements = read_measurements(f)
            result = count_increases(measurements)
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            measurements = read_measurements(f)
            window_sums = calc_window_sums(measurements, 3)
            result = count_increases(window_sums)
        return result


def read_measurements(fp: TextIO) -> Iterator[int]:
    return (int(line) for line in fp)


def calc_window_sums(measurements: Iterable[int], window_size: int) -> Iterator[int]:
    windows = create_windows(measurements, window_size)
    return (sum(window) for window in windows)


def create_windows(items: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    iterators = tee(items, n)
    offset_iterators = (islice(iterator, offset, None) for offset, iterator in enumerate(iterators))
    return zip(*offset_iterators)


def count_increases(measurements: Iterable[int]) -> int:
    return sum(
        1 for predecessor, successor in create_windows(measurements, 2) if successor > predecessor
    )


TEST_INPUT = """199
200
208
210
200
207
240
269
260
263
"""


@pytest.mark.parametrize(("window_size", "expected"), ((1, 7), (3, 5)))
def test_day01(window_size, expected):
    with StringIO(TEST_INPUT) as fp:
        measurements = read_measurements(fp)
        window_sums = calc_window_sums(measurements, window_size)
        result = count_increases(window_sums)

    assert expected == result
