"""Advent of Code 2015, day 13: https://adventofcode.com/2015/day/13"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import cached_property
from io import StringIO
from itertools import chain, permutations
from typing import IO, Iterator, TypeAlias

import pytest

from advent_of_code.base import Solution

GuestMatrix: TypeAlias = dict[frozenset[str], int]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            arranger = GuestArranger.load(f)
        arrangement = arranger.find_optimal_arrangement()
        print(arrangement)
        return arranger.calc_total_happiness_score(arrangement)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            arranger = GuestArranger.load(f, insert_self=True)
        arrangement = arranger.find_optimal_arrangement()
        print(arrangement)
        return arranger.calc_total_happiness_score(arrangement)


class GuestArranger:
    @classmethod
    def load(cls, f: IO, *args, **kwargs) -> GuestArranger:
        pattern = (
            r"(?P<guest1>\w+) would "  # First guest name
            r"(?P<direction>gain|lose) "  # Gain or loss
            r"(?P<magnitude>\d+) "  # Amount of happiness points
            r"happiness units by sitting next to "  # filler
            r"(?P<guest2>\w+)\."  # Second guest name
        )
        matrix: GuestMatrix = defaultdict(int)
        for line in f:
            properties = re.match(pattern, line).groupdict()
            key = frozenset([properties["guest1"], properties["guest2"]])
            vector = int(properties["magnitude"])
            if properties["direction"] == "lose":
                vector = -vector
            matrix[key] += vector
        return cls(matrix, *args, **kwargs)

    def __init__(self, matrix: GuestMatrix, insert_self: bool = False):
        self.matrix = matrix
        self.insert_self = insert_self

    @cached_property
    def guests(self) -> set[str]:
        result = set(chain.from_iterable(self.matrix.keys()))
        if self.insert_self:
            result.add("SELF")
        return result

    def generate_arrangements(self) -> Iterator[list[str]]:
        fixed_guest, *others = self.guests
        for others_arranged in permutations(others):
            yield [fixed_guest, *others_arranged]

    def calc_total_happiness_score(self, arrangement: list[str]) -> int:
        guests = arrangement + [arrangement[0]]
        pairs = (frozenset(p) for p in zip(guests, guests[1:]))
        return sum(self.matrix[pair] for pair in pairs)

    def find_optimal_arrangement(self) -> list[str]:
        return max(
            self.generate_arrangements(), key=lambda arr: self.calc_total_happiness_score(arr)
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
def guest_arranger(sample_input):
    return GuestArranger.load(sample_input)


class TestGuestArranger:
    def test_load(self, guest_arranger):
        expected_matrix = {
            frozenset(["Alice", "Bob"]): 54 + 83,
            frozenset(["Alice", "Carol"]): -79 - 62,
            frozenset(["Alice", "David"]): -2 + 46,
            frozenset(["Bob", "Carol"]): -7 + 60,
            frozenset(["Bob", "David"]): -63 - 7,
            frozenset(["Carol", "David"]): 55 + 41,
        }
        assert guest_arranger.matrix == expected_matrix

    def test_guests(self, guest_arranger):
        assert guest_arranger.guests == {"Alice", "Bob", "Carol", "David"}

    def test_generate_arrangements(self, guest_arranger):
        result = list(guest_arranger.generate_arrangements())
        assert len(result) == 6

    def calc_total_happiness_score(self, guest_arranger):
        arrangement = ["Alice", "Bob", "Carol", "David"]
        assert guest_arranger.calc_total_happiness_score(arrangement) == 330

    @pytest.mark.skip("Frozenset item order is not fixed, so this exact result is not determinate.")
    def test_find_optimal_arrangement(self, guest_arranger):
        assert guest_arranger.find_optimal_arrangement() == ["Carol", "David", "Alice", "Bob"]
