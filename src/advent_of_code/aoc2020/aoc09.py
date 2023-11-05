"""Advent of Code 2020, day 9: https://adventofcode.com/2020/day/9"""
from collections import deque
from io import StringIO
from itertools import islice
from typing import Iterable, Iterator, Sequence

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2020, **kwargs)

    def solve_part_one(self) -> int:
        decrypter = Decrypter(25)
        with self.open_input() as f:
            values = [int(n) for n in f.readlines()]
        return next(decrypter.find_invalid_values(values))

    def solve_part_two(self) -> int:
        decrypter = Decrypter(25)
        with self.open_input() as f:
            values = [int(n) for n in f.readlines()]
        return decrypter.calc_weakness(values)


class Decrypter:
    def __init__(self, preamble_length: int):
        self.preamble_length = preamble_length

    def find_invalid_values(self, values: Iterable[int]) -> Iterator[int]:
        values = iter(values)
        queue = deque()
        queue.extend(islice(values, self.preamble_length))

        def is_valid(n):
            for i, m in enumerate(queue):
                for o in islice(queue, i + 1):
                    if m + o == n:
                        return True
            return False

        for n in values:
            if not is_valid(n):
                yield n
            queue.append(n)
            queue.popleft()

    def calc_weakness(self, values: Sequence[int]) -> int:
        target_value = next(self.find_invalid_values(values))
        sum = 0
        queue = deque()
        for n in values:
            sum += n
            queue.append(n)
            while sum > target_value:
                sum -= queue.popleft()
            if sum == target_value:
                return min(queue) + max(queue)
        raise ValueError("No weakness found")


SAMPLE_INPUT = """\
35
20
15
25
47
40
62
55
65
95
102
117
150
182
127
219
299
277
309
576
"""


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_find_invalid_values(sample_input):
    decrypter = Decrypter(5)
    values = [int(n) for n in sample_input.readlines()]
    invalid_values = list(decrypter.find_invalid_values(values))
    assert invalid_values == [127]


def test_calc_weakness(sample_input):
    decrypter = Decrypter(5)
    values = [int(n) for n in sample_input.readlines()]
    weakness = decrypter.calc_weakness(values)
    assert weakness == 62
