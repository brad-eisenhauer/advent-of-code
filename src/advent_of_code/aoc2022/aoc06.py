"""Advent of Code 2022, day 6: https://adventofcode.com/2022/day/6"""

from __future__ import annotations

from io import StringIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            buffer = f.readline().strip()
        return find_marker(buffer)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            buffer = f.readline().strip()
        return find_marker(buffer, length=14)


def find_marker(buffer: str, length: int = 4) -> int:
    for i in range(length, len(buffer)):
        sub_str = buffer[i - length : i]
        if len(set(sub_str)) == length:
            return i


SAMPLE_INPUTS = [
    """\
mjqjpqmgbljsphdztnvjfqwrcgsmlb
""",
    """\
bvwbjplbgvbhsrlpgdmjqwftvncz
""",
    """\
nppdvjthqldpwncqszvftbrmjlhg
""",
    """\
nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg
""",
    """\
zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "length", "expected"),
    [
        (0, 4, 7),
        (1, 4, 5),
        (2, 4, 6),
        (3, 4, 10),
        (4, 4, 11),
        (0, 14, 19),
        (1, 14, 23),
        (2, 14, 23),
        (3, 14, 29),
        (4, 14, 26),
    ],
    indirect=["sample_input"],
)
def test_find_marker(sample_input, length, expected):
    buffer = sample_input.readline().strip()
    assert find_marker(buffer, length) == expected
