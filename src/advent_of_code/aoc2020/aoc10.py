"""Advent of Code 2020, day 10: https://adventofcode.com/2020/day/10"""
from collections import Counter
from functools import cache
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            jolts = list(read_jolts(f))
        ones, threes = calc_step_distribution(jolts)
        return ones * threes

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            jolts = tuple(sorted(read_jolts(f)))
        return count_configurations(jolts)


def read_jolts(f: TextIO) -> Iterator[int]:
    return (int(n) for n in f.readlines())


def calc_step_distribution(adapters: list[int]) -> tuple[int, int]:
    jolts = [0] + sorted(adapters) + [max(adapters) + 3]
    counts = Counter(b - a for a, b in zip(jolts, jolts[1:]))
    return counts[1], counts[3]


@cache
def count_configurations(adapters: tuple[int, ...], initial_joltage: int = 0) -> int:
    match adapters:
        case (last_adapter,):
            return 1 if (last_adapter - initial_joltage) <= 3 else 0
        case (next_adapter, *remainder) if (next_adapter - initial_joltage) <= 3:
            remainder = tuple(remainder)
            return count_configurations(remainder, next_adapter) + count_configurations(
                remainder, initial_joltage
            )
        case _:
            return 0


SAMPLE_INPUTS = [
    """\
16
10
15
5
1
11
7
19
6
12
4
""",
    """\
28
33
18
42
31
14
46
20
48
47
24
23
49
45
19
38
39
11
1
32
25
35
8
17
7
9
4
2
34
10
3
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, (7, 5)), (1, (22, 10))], indirect=["sample_input"]
)
def test_calc_step_distribution(sample_input, expected):
    jolts = list(read_jolts(sample_input))
    assert calc_step_distribution(jolts) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 8), (1, 19208)], indirect=["sample_input"]
)
def test_count_configurations(sample_input, expected):
    jolts = tuple(sorted(read_jolts(sample_input)))
    assert count_configurations(jolts) == expected
