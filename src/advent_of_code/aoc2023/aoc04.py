"""Advent of Code 2023, day 4: https://adventofcode.com/2023/day/4"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Sequence

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2023, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            return sum(ScratchCard.read(line.strip()).calc_point_value() for line in fp)

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            all_cards = [ScratchCard.read(line.strip()) for line in fp]
        return calc_total_cards_won(all_cards)


@dataclass
class ScratchCard:
    winning_numbers: set[int]
    card_numbers: set[int]

    @classmethod
    def read(cls, text: str) -> ScratchCard:
        _, numbers = text.split(": ")
        winning_number_str, card_number_str = numbers.split(" | ")
        winning_numbers = {int(n) for n in winning_number_str.split()}
        card_numbers = {int(n) for n in card_number_str.split()}
        return ScratchCard(winning_numbers, card_numbers)

    def calc_point_value(self) -> int:
        common_number_count = self.calc_common_number_count()
        if common_number_count < 1:
            return 0
        return 2 ** (common_number_count - 1)

    def calc_common_number_count(self) -> int:
        return len(self.winning_numbers & self.card_numbers)


def calc_total_cards_won(all_cards: Sequence[ScratchCard]) -> int:
    copy_count = [1] * len(all_cards)
    for i, card in enumerate(all_cards):
        for j in range(card.calc_common_number_count()):
            copy_count[i + j + 1] += copy_count[i]
    log.debug("%s", copy_count)
    return sum(copy_count)


SAMPLE_INPUTS = [
    """\
Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11
""",
]


@pytest.fixture
def sample_input(request: pytest.FixtureRequest) -> Iterator[IO]:
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


def test_calc_point_value(sample_input: IO):
    result = [ScratchCard.read(line.strip()).calc_point_value() for line in sample_input]
    assert result == [8, 2, 2, 1, 0, 0]


def test_calc_total_card_count(sample_input: IO):
    cards = [ScratchCard.read(line.strip()) for line in sample_input]
    assert calc_total_cards_won(cards) == 30
