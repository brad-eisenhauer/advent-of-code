"""Advent of Code 2023, day 7: https://adventofcode.com/2023/day/7"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from io import StringIO
from itertools import count
from typing import IO, Iterable, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            hands = sorted(Hand.read(line) for line in fp)
        return self._calc_score(hands)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            hands = sorted(Hand.read(line, j_is_wild=True) for line in fp)
        return self._calc_score(hands)

    @staticmethod
    def _calc_score(hands: Iterable[Hand]) -> int:
        return sum(rank * hand.bid for rank, hand in zip(count(1), hands))


class HandType(IntEnum):
    HighCard = 1
    OnePair = 2
    TwoPairs = 3
    ThreeOfAKind = 4
    FullHouse = 5
    FourOfAKind = 6
    FiveOfAKind = 7


@dataclass(frozen=True)
class Hand:
    cards: str
    bid: int
    j_is_wild: bool = False

    @classmethod
    def read(cls, text: str, j_is_wild: bool = False) -> Hand:
        cards, bid_str = text.split()
        return Hand(cards, int(bid_str), j_is_wild)

    @cached_property
    def type(self) -> HandType:
        card_counts = Counter(self.cards)
        sorted_counts = sorted(card_counts.values(), reverse=True)
        log.debug("card=%s, counts=%s", self, card_counts)
        match sorted_counts:
            case [5]:
                return HandType.FiveOfAKind
            case [4, 1] if self.j_is_wild and card_counts.get("J", 0) > 0:
                return HandType.FiveOfAKind
            case [4, 1]:
                return HandType.FourOfAKind
            case [3, 2] if self.j_is_wild and card_counts.get("J", 0) > 0:
                return HandType.FiveOfAKind
            case [3, 2]:
                return HandType.FullHouse
            case [3, *_] if self.j_is_wild and card_counts.get("J", 0) > 0:
                return HandType.FourOfAKind
            case [3, *_]:
                return HandType.ThreeOfAKind
            case [2, 2, 1] if self.j_is_wild and (joker_count := card_counts.get("J", 0)) > 0:
                return HandType.FourOfAKind if joker_count == 2 else HandType.FullHouse
            case [2, 2, 1]:
                return HandType.TwoPairs
            case [2, *_] if self.j_is_wild and card_counts.get("J", 0) > 0:
                return HandType.ThreeOfAKind
            case [2, *_]:
                return HandType.OnePair
            case _ if self.j_is_wild and card_counts.get("J", 0) > 0:
                return HandType.OnePair
            case _:
                return HandType.HighCard

    @cached_property
    def card_values(self) -> tuple[int, ...]:
        values = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 1 if self.j_is_wild else 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
        return tuple(values[c] for c in self.cards)

    def __lt__(self, other) -> bool:
        if not isinstance(other, Hand):
            raise TypeError()
        return (self.type, self.card_values) < (other.type, other.card_values)


SAMPLE_INPUTS = [
    """\
32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 6440


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 5905


class TestHand:
    def test_read(self, sample_input):
        result = [Hand.read(line) for line in sample_input]
        assert result == [
            Hand("32T3K", 765),
            Hand("T55J5", 684),
            Hand("KK677", 28),
            Hand("KTJJT", 220),
            Hand("QQQJA", 483),
        ]

    @pytest.mark.parametrize(
        ("j_as_joker", "expected"),
        [
            (
                False,
                [
                    HandType.OnePair,
                    HandType.ThreeOfAKind,
                    HandType.TwoPairs,
                    HandType.TwoPairs,
                    HandType.ThreeOfAKind,
                ],
            ),
            (
                True,
                [
                    HandType.OnePair,
                    HandType.FourOfAKind,
                    HandType.TwoPairs,
                    HandType.FourOfAKind,
                    HandType.FourOfAKind,
                ],
            ),
        ],
    )
    def get_type(self, sample_input, j_as_joker, expected):
        result = [Hand.read(line, j_as_joker).type for line in sample_input]
        assert result == expected

    @pytest.mark.parametrize(
        ("hand", "expected"),
        [
            (Hand("9889J", 0), HandType.TwoPairs),
            (Hand("9889J", 0, j_is_wild=True), HandType.FullHouse),
            (Hand("J37QA", 0), HandType.HighCard),
            (Hand("J37QA", 0, j_is_wild=True), HandType.OnePair),
            (Hand("JTJ63", 0), HandType.OnePair),
            (Hand("JTJ63", 0, j_is_wild=True), HandType.ThreeOfAKind),
            (Hand("2QQ5J", 0), HandType.OnePair),
            (Hand("2QQ5J", 0, j_is_wild=True), HandType.ThreeOfAKind),
            (Hand("JJJJ1", 0), HandType.FourOfAKind),
            (Hand("JJJJ1", 0, j_is_wild=True), HandType.FiveOfAKind),
            (Hand("JQJQQ", 0), HandType.FullHouse),
            (Hand("JQJQQ", 0, j_is_wild=True), HandType.FiveOfAKind),
            (Hand("JJJ12", 0), HandType.ThreeOfAKind),
            (Hand("JJJ12", 0, j_is_wild=True), HandType.FourOfAKind),
        ],
    )
    def test_type_with_j_as_wild(self, hand: Hand, expected: HandType):
        assert hand.type == expected

    def test_ordering(self, sample_input):
        result = sorted(Hand.read(line) for line in sample_input)
        log.debug(result)
        assert result == [
            Hand("32T3K", 765),
            Hand("KTJJT", 220),
            Hand("KK677", 28),
            Hand("T55J5", 684),
            Hand("QQQJA", 483),
        ]
