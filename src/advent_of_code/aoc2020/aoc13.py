"""Advent of Code 2020, day 13: https://adventofcode.com/2020/day/13"""
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import least_common_multiple


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            timestamp, buses = read_input(f)
        wait, bus = calc_earliest_departure(timestamp, buses)
        return wait * bus

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            _, buses = read_input(f, read_all=True)
        result, _ = calc_earliest_consecutive_departures(buses)
        return result


def read_input(f: TextIO, read_all: bool = False) -> tuple[int, list[Optional[int]]]:
    timestamp = int(f.readline().strip())
    buses = [int(n) if n.isnumeric() else None for n in f.readline().strip().split(",")]
    if not read_all:
        buses = [b for b in buses if b is not None]
    return timestamp, buses


def calc_earliest_departure(timestamp: int, buses: list[int]) -> tuple[int, int]:
    min_wait = min(buses)
    min_wait_bus: Optional[int] = None
    for b in buses:
        this_wait = b - (timestamp % b)
        if min_wait_bus is None or this_wait < min_wait:
            min_wait = this_wait
            min_wait_bus = b
    return min_wait, min_wait_bus


def calc_earliest_consecutive_departures(buses: list[Optional[int]]) -> tuple[int, int]:
    """
    Calculate the earliest timestamp t at which buses begin departing according to the list

    For example [17, x, 13, 19] we're looking for t such that:
    - t % 17 == 0
    - (t + 2) % 13 == 0
    - (t + 3) % 19 == 0

    More generally, for each element b_i in b (at index i): (t + i) % b_i == 0.

    Parameters
    ----------
    buses: list[Optional[int]]
        Bus IDs

    Returns
    -------
    int, int
        - earliest timestamp at which buses begin leaving the port consecutively
        - interval at which the same departure pattern will repeat
    """
    # No inputs start with an "x", otherwise we'd need error trapping around this.
    while buses[-1] is None:
        del buses[-1]
    if len(buses) == 1:
        return buses[0], buses[0]
    ts, interval = calc_earliest_consecutive_departures(buses[:-1])
    bus_interval = buses[-1]
    # assert greatest_common_divisor(interval, bus_interval) == 1
    while True:
        if (ts + len(buses) - 1) % bus_interval == 0:
            return ts, least_common_multiple(interval, bus_interval)
        ts += interval


SAMPLE_INPUTS = [
    """\
939
7,13,x,x,59,x,31,19
""",
    """\
0
17,x,13,19
""",
    """\
0
67,7,59,61
""",
    """\
0
67,x,7,59,61
""",
    """\
0
67,7,x,59,61
""",
    """\
0
1789,37,47,1889
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_calc_earliest_departure(sample_input):
    timestamp, buses = read_input(sample_input)
    wait, bus = calc_earliest_departure(timestamp, buses)
    assert wait * bus == 295


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 1068781), (1, 3417), (2, 754018), (3, 779210), (4, 1261476), (5, 1202161486)],
    indirect=["sample_input"],
)
def test_calc_earliest_consecutive_departures(sample_input, expected):
    _, buses = read_input(sample_input, read_all=True)
    result, _ = calc_earliest_consecutive_departures(buses)
    assert result == expected
