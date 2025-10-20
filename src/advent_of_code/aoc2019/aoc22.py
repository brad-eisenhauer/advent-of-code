"""Advent of Code 2019, day 22: https://adventofcode.com/2019/day/22"""

import logging
from dataclasses import dataclass, field
from io import StringIO
from itertools import takewhile
from typing import Iterable

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import mod_inverse

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(22, 2019, **kwargs)

    def solve_part_one(self) -> int:
        deck = Deck(10007)
        with self.open_input() as f:
            deck.shuffle(f.readlines())
        return deck.index_of(2019)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            instructions = f.readlines()
        deck = Deck(119315717514047)
        deck.shuffle(instructions)
        deck.repeat(101741582076661)
        return deck.value_of(2020)


@dataclass
class Deck(Iterable[int]):
    """State of the deck

    Attributes
    ----------
    size: int
        Number of cards in the deck
    top: int
        Value of the top card in the deck
    interval: int
        Difference in value between consecutive cards in the deck
    """

    size: int
    top: int = field(default=0, init=False)
    interval: int = field(default=1, init=False)

    def __iter__(self):
        for n in range(self.size):
            yield (self.top + n * self.interval) % self.size

    @property
    def bottom(self) -> int:
        return (self.top - self.interval) % self.size

    def shuffle(self, instructions: Iterable[str]):
        for instruction in instructions:
            if instruction.startswith("cut"):
                self.cut(int(instruction.split()[-1]))
            elif "with increment" in instruction:
                self.deal_with_increment(int(instruction.split()[-1]))
            elif "new stack" in instruction:
                self.reverse()

    def cut(self, increment: int):
        self.top = (self.top + increment * self.interval) % self.size

    def reverse(self):
        self.top = self.bottom
        self.interval = (self.interval * -1) % self.size

    def deal_with_increment(self, increment: int):
        self.interval = (self.interval * mod_inverse(increment, self.size)) % self.size

    def repeat(self, n: int):
        """Repeat the shuffles already applied

        Parameters
        ----------
        n: int
            Number of times to repeat the shuffle. n=1 leaves the deck as-is.
        """
        result_top = 0
        result_interval = 1
        while n > 0:
            if n % 2 == 1:
                result_top = (result_top + self.top * result_interval) % self.size
                result_interval = (result_interval * self.interval) % self.size
            n >>= 1
            self.top = (self.top + self.top * self.interval) % self.size
            self.interval = (self.interval * self.interval) % self.size
        self.top = result_top
        self.interval = result_interval

    def index_of(self, value: int) -> int:
        result = value
        result -= self.top
        result *= mod_inverse(self.interval, self.size)
        result %= self.size
        return result

    def value_of(self, index: int) -> int:
        return (self.top + (index * self.interval)) % self.size


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


@pytest.mark.parametrize("sample_input", list(range(len(SAMPLE_INPUTS))), indirect=True)
@pytest.mark.parametrize("size", [10, 16, 20, 25, 32, 50, 64, 100])
@pytest.mark.parametrize("target", list(range(10)))
def test_index_of(sample_input, size, target):
    deck = Deck(size)
    deck.shuffle(sample_input.readlines())
    expected = sum(1 for _ in takewhile(lambda n: n != target, deck))
    assert deck.index_of(target) == expected


@pytest.mark.parametrize("sample_input", list(range(len(SAMPLE_INPUTS))), indirect=True)
@pytest.mark.parametrize("size", [10, 16, 20, 25, 32])
@pytest.mark.parametrize("n", [1, 5, 10, 20, 50, 100, 250, 500, 1025])
def test_repeat(sample_input, size, n):
    instructions = sample_input.readlines()
    expected = Deck(size)
    for _ in range(n):
        expected.shuffle(instructions)
    deck = Deck(size)
    deck.shuffle(instructions)
    deck.repeat(n)
    assert deck == expected
