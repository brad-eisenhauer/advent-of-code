"""Advent of Code 2019, day 16: https://adventofcode.com/2019/day/16"""

from functools import cache
from itertools import chain, islice, repeat
from typing import Iterable

import numpy as np

from advent_of_code.base import Solution

BASE_PATTERN = [0, 1, 0, -1]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            result = [int(n) for n in f.read().strip()]
        for _ in range(100):
            result = calc_next_phase(result)
        return int("".join(str(n) for n in result[:8]))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            initial_value = f.read().strip()
        start_index = int(initial_value[:7])
        assert start_index > (len(initial_value) * 10000) // 2
        result = [int(c) for c in (initial_value * 10000)[start_index:]]
        for _ in range(100):
            result = calc_back_half(result)
        return int("".join(str(n) for n in result[:8]))


@cache
def create_pattern_matrix(dim: int) -> np.ndarray:
    def make_row(n) -> Iterable[int]:
        duped_pattern = list(chain.from_iterable(zip(*[BASE_PATTERN] * (n + 1))))
        row = islice(chain.from_iterable(repeat(duped_pattern)), 1, dim + 1)
        return row

    result = np.fromiter(
        chain.from_iterable(make_row(n) for n in range(dim)),
        dtype=int,
        count=dim * dim,
    )
    return result.reshape((dim, dim))


def calc_next_phase(nums: list[int]) -> list[int]:
    dim = len(nums)
    matrix = create_pattern_matrix(dim)
    expanded = np.array(nums * dim).reshape((dim, dim))
    result = np.abs((matrix * expanded).sum(axis=1)) % 10
    return list(result)


def calc_back_half(nums: list[int]) -> list[int]:
    result = []
    acc = 0
    for n in nums[::-1]:
        acc = (acc + n) % 10
        result.append(acc)
    return list(reversed(result))


def test_create_pattern_matrix():
    expected = np.array(
        [
            [1, 0, -1, 0, 1, 0, -1, 0],
            [0, 1, 1, 0, 0, -1, -1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    assert (create_pattern_matrix(8) == expected).all()


def test_calc_next_phase():
    nums = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = [4, 8, 2, 2, 6, 1, 5, 8]
    assert calc_next_phase(nums) == expected
