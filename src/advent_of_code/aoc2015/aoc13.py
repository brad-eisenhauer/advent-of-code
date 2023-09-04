"""Advent of Code 2015, day 13: https://adventofcode.com/2015/day/13"""
from __future__ import annotations

import re
from collections import defaultdict
from io import StringIO
from itertools import permutations
from typing import Iterable, Iterator, TypeAlias

import pytest

from advent_of_code.base import Solution

GuestMatrix: TypeAlias = dict[tuple[str, str], int]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            matrix = create_score_matrix(f)
        arrangement = find_optimal_arrangement(matrix)
        return calc_total_happiness_score(arrangement, matrix)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            matrix = create_score_matrix(f)
        arrangement = find_optimal_arrangement(matrix, insert_self=True)
        return calc_total_happiness_score(arrangement, matrix)


def create_score_matrix(f: StringIO) -> GuestMatrix:
    pattern = re.compile(
        r"(?P<name_1>\w+) would (?P<sign>gain|lose) (?P<quantity>\d+) "
        r"happiness units by sitting next to (?P<name_2>\w+)\."
    )
    result = defaultdict(int)
    for line in f:
        properties = pattern.match(line).groupdict()
        quantity = int(properties["quantity"])
        if properties["sign"] == "lose":
            quantity = -quantity
        result[(properties["name_1"], properties["name_2"])] = quantity
    return result


def extract_guest_list(matrix: GuestMatrix) -> set[str]:
    result = set()
    for name, _ in matrix.keys():
        result.add(name)
    return result


def calc_total_happiness_score(arrangement: list[str], matrix: GuestMatrix) -> int:
    arrangement = arrangement + [arrangement[0]]
    pairs = zip(arrangement, arrangement[1:])
    result = sum(matrix[pair] for pair in pairs)
    arrangement = list(reversed(arrangement))
    pairs = zip(arrangement, arrangement[1:])
    result += sum(matrix[pair] for pair in pairs)
    return result


def generate_arrangements(guests: Iterable[str]) -> Iterator[list[str]]:
    guest_1, *others = guests
    for others_arranged in permutations(others):
        yield [guest_1, *others_arranged]


def find_optimal_arrangement(matrix: GuestMatrix, insert_self: bool = False) -> int:
    guests = extract_guest_list(matrix)
    if insert_self:
        guests = [None, *guests]
    return max(
        generate_arrangements(guests),
        key=lambda arrangement: calc_total_happiness_score(arrangement, matrix),
    )


SAMPLE_INPUTS = [
    """\
Alice would gain 54 happiness units by sitting next to Bob.
Alice would lose 79 happiness units by sitting next to Carol.
Alice would lose 2 happiness units by sitting next to David.
Bob would gain 83 happiness units by sitting next to Alice.
Bob would lose 7 happiness units by sitting next to Carol.
Bob would lose 63 happiness units by sitting next to David.
Carol would lose 62 happiness units by sitting next to Alice.
Carol would gain 60 happiness units by sitting next to Bob.
Carol would gain 55 happiness units by sitting next to David.
David would gain 46 happiness units by sitting next to Alice.
David would lose 7 happiness units by sitting next to Bob.
David would gain 41 happiness units by sitting next to Carol.
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture
def matrix(sample_input):
    return create_score_matrix(sample_input)


def test_create_score_matrix(sample_input):
    result = create_score_matrix(sample_input)
    expected = {
        ("Alice", "Bob"): 54,
        ("Alice", "Carol"): -79,
        ("Alice", "David"): -2,
        ("Bob", "Alice"): 83,
        ("Bob", "Carol"): -7,
        ("Bob", "David"): -63,
        ("Carol", "Alice"): -62,
        ("Carol", "Bob"): 60,
        ("Carol", "David"): 55,
        ("David", "Alice"): 46,
        ("David", "Bob"): -7,
        ("David", "Carol"): 41,
    }
    assert result == expected


def test_extract_guest_list(matrix):
    result = extract_guest_list(matrix)
    assert result == {"Alice", "Bob", "Carol", "David"}


def test_calc_total_happiness_score(matrix):
    guests = ["Alice", "Bob", "Carol", "David"]
    assert calc_total_happiness_score(guests, matrix) == 330


def test_generate_arrangements():
    guests = ["Alice", "Bob", "Carol", "David"]
    expected = [
        ["Alice", "Bob", "Carol", "David"],
        ["Alice", "Bob", "David", "Carol"],
        ["Alice", "Carol", "Bob", "David"],
        ["Alice", "Carol", "David", "Bob"],
        ["Alice", "David", "Bob", "Carol"],
        ["Alice", "David", "Carol", "Bob"],
    ]
    assert list(generate_arrangements(guests)) == expected


def test_find_optimal_arrangement(matrix: GuestMatrix):
    assert find_optimal_arrangement(matrix) == ["Carol", "David", "Alice", "Bob"]
