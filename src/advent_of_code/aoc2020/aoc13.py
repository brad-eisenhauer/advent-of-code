"""Advent of Code 2020, day 13: https://adventofcode.com/2020/day/13"""
from io import StringIO
from typing import Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(13, 2020)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            timestamp = int(f.readline().strip())
            buses = [int(n) for n in f.readline().strip().split(",") if n != "x"]
        wait, bus = calc_earliest_departure(timestamp, buses)
        return wait * bus


def calc_earliest_departure(timestamp: int, buses: list[int]) -> tuple[int, int]:
    min_wait = min(buses)
    min_wait_bus: Optional[int] = None
    for b in buses:
        this_wait = b - (timestamp % b)
        if min_wait_bus is None or this_wait < min_wait:
            min_wait = this_wait
            min_wait_bus = b
    return min_wait, min_wait_bus


def calc_earliest_consecutive_departures(buses: list[Optional[int]]) -> int:
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
        ...

    Returns
    -------
    int
        earliest timestamp at which buses begin leaving the port consecutively
    """
    ...


SAMPLE_INPUTS = [
    """\
939
7,13,x,x,59,x,31,19
""",
    """\

17,x,13,19
""",
    """\

67,7,59,61
""",
    """\

67,x,7,59,61
""",
    """\

67,7,x,59,61
""",
    """\

1789,37,47,1889
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_calc_earliest_departure(sample_input):
    timestamp = int(sample_input.readline().strip())
    buses = [int(n) for n in sample_input.readline().strip().split(",") if n != "x"]
    wait, bus = calc_earliest_departure(timestamp, buses)
    assert wait * bus == 295


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 1068781), (1, 3417), (2, 754018), (3, 779210), (4, 1261476), (5, 1202161486)],
    indirect=["sample_input"],
)
def test_calc_earliest_consecutive_departures(sample_input, expected):
    _ = sample_input.readline()
    buses = [int(n) if n.isnumeric() else None for n in sample_input.readline().strip().split(",")]
    assert calc_earliest_consecutive_departures(buses) == expected
