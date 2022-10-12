"""Advent of Code 2020, day 10: https://adventofcode.com/2020/day/10"""
from collections import Counter
from functools import cache
from io import StringIO
from typing import TextIO, Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(10, 2020)

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


def calc_step_distribution(jolts: list[int]) -> tuple[int, int]:
    jolts = sorted(jolts + [0])
    jolts.append(jolts[-1] + 3)
    counts = Counter(b - a for a, b in zip(jolts, jolts[1:]))
    return counts[1], counts[3]


@cache
def count_configurations(adapters: tuple[int, ...], initial_voltage: int = 0) -> int:
    match adapters:
        case (last_adapter,):
            return 1 if (last_adapter - initial_voltage) <= 3 else 0
        case (next_adapter, *remainder) if (next_adapter - initial_voltage) <= 3:
            remainder = tuple(remainder)
            return count_configurations(remainder, next_adapter) + count_configurations(remainder, initial_voltage)
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
"""
]


@pytest.fixture(params=range(len(SAMPLE_INPUTS)))
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("input_index", "expected"), [(0, (7, 5)), (1, (22, 10))]
)
def test_calc_step_distribution(input_index, expected):
    with StringIO(SAMPLE_INPUTS[input_index]) as f:
        jolts = read_jolts(f)
    assert calc_step_distribution(jolts) == expected


@pytest.mark.parametrize(
    ("input_index", "expected"), [(0, 8), (1, 19208)]
)
def test_count_configurations(input_index, expected):
    with StringIO(SAMPLE_INPUTS[input_index]) as f:
        jolts = tuple(sorted(read_jolts(f)))
    assert count_configurations(jolts) == expected
