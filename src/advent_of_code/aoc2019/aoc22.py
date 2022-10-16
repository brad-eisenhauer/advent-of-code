"""Advent of Code 2019, day 22: https://adventofcode.com/2019/day/22"""
import logging
from dataclasses import dataclass
from io import StringIO
from itertools import count
from typing import Iterable

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(22, 2019)

    def solve_part_one(self) -> int:
        deck = Deck(10007)
        with self.open_input() as f:
            deck.shuffle(f.readlines())
        for instruction in deck.equivalent():
            log.debug(instruction)
        return deck.index_of(2019)

    def solve_part_two(self) -> int:
        deck = Deck(119315717514047)
        with self.open_input() as f:
            instructions = f.readlines()
        deck.shuffle(instructions)
        for inst in deck.equivalent():
            log.debug(inst)
        # deck.shuffle(chain.from_iterable(repeat(deck.equivalent(), 101741582076661 - 1)))
        # return deck.index_of(2020)


@dataclass
class Deck(Iterable[int]):
    size: int

    def __post_init__(self):
        self.top = 0
        self.deal_interval = 1

    def __iter__(self):
        for n in range(self.size):
            yield (self.top + n * self.deal_interval) % self.size

    @property
    def bottom(self) -> int:
        return (self.top - self.deal_interval) % self.size

    def cut(self, increment: int):
        self.top += increment * self.deal_interval
        self.top %= self.size

    def reverse(self):
        self.top = self.bottom
        self.deal_interval *= -1
        self.deal_interval %= self.size

    def deal_with_increment(self, increment: int):
        # find the smallest n such that (n * increment % size) == 1
        self.deal_interval *= self.calc_complementary_interval(increment)
        self.deal_interval %= self.size

    def index_of(self, value: int) -> int:
        result = value
        result -= self.top
        result *= self.calc_complementary_interval(self.deal_interval)
        result %= self.size
        return result

    def shuffle(self, instructions: Iterable[str]):
        for instruction in instructions:
            if instruction.startswith("cut"):
                self.cut(int(instruction.split()[-1]))
            elif "with increment" in instruction:
                self.deal_with_increment(int(instruction.split()[-1]))
            elif "new stack" in instruction:
                self.reverse()

    def calc_complementary_interval(self, increment: int) -> int:
        for n in count(1):
            if (n * self.size + 1) % increment == 0:
                return (n * self.size + 1) // increment

    def equivalent(self) -> list[str]:
        result = []
        if self.top:
            result.append(f"cut {self.top}")
        if self.deal_interval > 1:
            result.append(
                f"deal with increment {self.calc_complementary_interval(self.deal_interval)}"
            )
        return result


SAMPLE_INPUTS = [
    """\
deal with increment 7
deal into new stack
deal into new stack
""",
    """\
cut 6
deal with increment 7
deal into new stack
""",
    """\
deal with increment 7
deal with increment 9
cut -2
""",
    """\
deal into new stack
cut -2
deal with increment 7
cut 8
cut -4
deal with increment 7
cut 3
deal with increment 9
deal with increment 3
cut -1
""",
    """\
deal into new stack
""",
    """\
deal with increment 7
""",
    """\
cut 4
""",
    """\
cut -4
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "size", "expected"),
    [
        (0, 10, [0, 3, 6, 9, 2, 5, 8, 1, 4, 7]),
        (1, 10, [3, 0, 7, 4, 1, 8, 5, 2, 9, 6]),
        (2, 10, [6, 3, 0, 7, 4, 1, 8, 5, 2, 9]),
        (3, 10, [9, 2, 5, 8, 1, 4, 7, 0, 3, 6]),
        (4, 10, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        (5, 10, [0, 3, 6, 9, 2, 5, 8, 1, 4, 7]),
        (6, 10, [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
        (7, 10, [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]),
        (5, 15, [0, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2]),
    ],
    indirect=["sample_input"],
)
def test_shuffle(sample_input, size, expected):
    instructions = sample_input.readlines()
    deck = Deck(size)
    deck.shuffle(instructions)
    assert list(deck) == expected


@pytest.mark.parametrize(
    ("sample_input", "size"),
    [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (5, 15)],
    indirect=["sample_input"],
)
def test_equivalent(sample_input, size):
    deck1 = Deck(size)
    deck1.shuffle(sample_input.readlines())
    deck2 = Deck(size)
    deck2.shuffle(deck1.equivalent())
    assert list(deck2) == list(deck1)


@pytest.mark.parametrize(
    ("sample_input", "size", "expected"),
    [
        (0, 10, 1),
        (1, 10, 0),
        (2, 10, 1),
        (3, 10, 8),
        (4, 10, 6),
        (5, 10, 1),
        (6, 10, 9),
        (7, 10, 7),
        (5, 15, 6),
    ],
    indirect=["sample_input"],
)
def test_index_of(sample_input, size, expected):
    deck = Deck(size)
    deck.shuffle(sample_input.readlines())
    assert deck.index_of(3) == expected
