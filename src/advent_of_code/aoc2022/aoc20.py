"""Advent of Code 2022, day 20: https://adventofcode.com/2022/day/20"""
from __future__ import annotations

from io import StringIO
from typing import TextIO, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(20, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            values = read(f)
        decrypted = decrypt(values)
        return calc_sum_of_coords(decrypted)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            values = read(f, key=811589153)
        decrypted = decrypt(values, n=10)
        return calc_sum_of_coords(decrypted)


def read(f: TextIO, key: Optional[int] = None) -> list[tuple[int, int]]:
    return [(i, int(n) * key if key is not None else int(n)) for i, n in enumerate(f)]


def decrypt(values: list[tuple[int, int]], n: int = 1) -> list[int]:
    result = values.copy()
    for _ in range(n):
        for value in values:
            i = result.index(value)
            j = (i + value[1]) % (len(result) - 1)
            del result[i]
            result.insert(j, value)
    return [v[1] for v in result]


def calc_sum_of_coords(decrypted: list[int]) -> int:
    i = decrypted.index(0)
    return sum(decrypted[(i + offset) % len(decrypted)] for offset in [1000, 2000, 3000])


SAMPLE_INPUTS = [
    """\
1
2
-3
3
-2
0
4
""",
    """\
1
2
-3
3
-2
0
4
20
""",
    """\
1
2
-3
3
-2
0
6
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, [-2, 1, 2, -3, 4, 0, 3]),
        (1, [-2, 1, 2, 4, 0, -3, 20, 3]),
        (2, [6, -2, 1, 2, -3, 0, 3]),
    ],
    indirect=["sample_input"],
)
def test_decrypt(sample_input, expected):
    values = read(sample_input)
    assert decrypt(values) == expected


@pytest.mark.parametrize(("sample_input", "expected"), [(0, 3), (1, 0)], indirect=["sample_input"])
def test_calc_sum_of_coords(sample_input, expected):
    values = read(sample_input)
    decrypted = decrypt(values)
    assert calc_sum_of_coords(decrypted) == expected
