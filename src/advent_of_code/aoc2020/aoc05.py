"""Advent of Code 2020, day 5: https://adventofcode.com/2020/day/5"""
import operator
from functools import reduce
from itertools import chain

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(5, 2020)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return max(calc_seat_id(t) for t in f)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            min_seat_id = 1024
            max_seat_id = 0
            result = 0
            for line in f:
                seat_id = calc_seat_id(line)
                min_seat_id = min(min_seat_id, seat_id)
                max_seat_id = max(max_seat_id, seat_id)
                result = result ^ seat_id
            result = reduce(
                operator.xor, chain(range(0, min_seat_id), range(max_seat_id + 1, 1024)), result
            )
            return result


def calc_row(ticket: str) -> int:
    return calc_seat_id(ticket[:7])


def calc_seat(ticket: str) -> int:
    return calc_seat_id(ticket[7:])


def calc_seat_id(ticket: str) -> int:
    result = 0
    for c in ticket.strip():
        value = 1 if c in "BR" else 0
        result *= 2
        result += value
    return result


@pytest.mark.parametrize(
    ("ticket", "expected"),
    [
        ("FBFBBFFRLR", 44),
        ("BFFFBBFRRR", 70),
        ("FFFBBBFRRR", 14),
        ("BBFFBBFRLL", 102),
    ],
)
def test_calc_row(ticket, expected):
    assert calc_row(ticket) == expected


@pytest.mark.parametrize(
    ("ticket", "expected"),
    [
        ("FBFBBFFRLR", 5),
        ("BFFFBBFRRR", 7),
        ("FFFBBBFRRR", 7),
        ("BBFFBBFRLL", 4),
    ],
)
def test_calc_row(ticket, expected):
    assert calc_seat(ticket) == expected


@pytest.mark.parametrize(
    ("ticket", "expected"),
    [
        ("FBFBBFFRLR", 357),
        ("BFFFBBFRRR", 567),
        ("FFFBBBFRRR", 119),
        ("BBFFBBFRLL", 820),
    ],
)
def test_calc_seat_id(ticket, expected):
    assert calc_seat_id(ticket) == expected
