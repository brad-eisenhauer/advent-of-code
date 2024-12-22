"""Advent of Code 2024, day 22: https://adventofcode.com/2024/day/22"""

from __future__ import annotations

from collections import defaultdict
from io import StringIO
from itertools import islice
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(22, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                seed = int(line)
                result += next(islice(prng(seed), 1999, None))
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        total_purchased_bananas: dict[int, int] = defaultdict(int)
        with input_file or self.open_input() as reader:
            for line in reader:
                seed = int(line)
                banana_counts = calc_purchased_bananas(seed)
                for change_index, banana_count in banana_counts.items():
                    total_purchased_bananas[change_index] += banana_count
        return max(total_purchased_bananas.values())


def prng(seed: int) -> Iterator[int]:
    secret = seed

    def _mix(n: int) -> int:
        return n ^ secret

    def _prune() -> int:
        return secret % 16777216

    while True:
        secret = _mix(secret << 6)
        secret = _prune()
        secret = _mix(secret >> 5)
        secret = _prune()
        secret = _mix(secret << 11)
        secret = _prune()
        yield secret


def calc_prices(seed: int) -> Iterator[int]:
    for n in prng(seed):
        yield n % 10


def calc_purchased_bananas(seed: int) -> dict[int, int]:
    result: dict[int, int] = {}
    prices = islice(calc_prices(seed), 2000)
    index: int = 0

    def _calc_index(change: int) -> int:
        return ((index << 5) + change + 9) & (2**20 - 1)

    last_price = next(prices)
    for price in islice(prices, 3):
        index = _calc_index(price - last_price)
        last_price = price
    for price in prices:
        index = _calc_index(price - last_price)
        if index not in result:
            result[index] = price
        last_price = price
    return result


SAMPLE_INPUTS = [
    """\
1
10
100
2024
""",
    """\
1
2
3
2024
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_prng() -> None:
    assert list(islice(prng(123), 10)) == [
        15887950,
        16495136,
        527345,
        704524,
        1553684,
        12683156,
        11100544,
        12249484,
        7753432,
        5908254,
    ]


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 37327623


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 23
