"""Advent of Code 2020, day 22: https://adventofcode.com/2020/day/22"""
from __future__ import annotations

from collections import deque
from io import StringIO
from itertools import count, takewhile
from typing import Iterable, TextIO, Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(22, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            decks = read_decks(f)
        winning_deck, _ = run_game(*decks)
        return winning_deck.calc_score()


class Deck(Iterator[int]):
    def __init__(self, name: str, cards: Iterable[int]):
        self.name = name
        self.cards = deque(cards)

    @classmethod
    def read(cls, f: TextIO) -> Deck:
        name = f.readline().rstrip("\n:")
        cards = (int(line.strip()) for line in takewhile(lambda l: l.strip(), f))
        return cls(name, cards)

    def __next__(self):
        return self.cards.popleft()

    def __eq__(self, other):
        return self.name == other.name and list(self.cards) == list(other.cards)

    def __repr__(self):
        return f"Deck('{self.name}', [{', '.join(str(n) for n in self.cards)}])"

    def append(self, *cards: int):
        self.cards.extend(cards)

    def calc_score(self) -> int:
        return sum(i * c for i, c in zip(count(1), reversed(self.cards)))


def read_decks(f: TextIO) -> tuple[Deck, Deck]:
    deck1 = Deck.read(f)
    deck2 = Deck.read(f)
    return deck1, deck2


def run_game(deck1: Deck, deck2: Deck) -> tuple[Deck, int]:
    battle_count = 0
    while deck1.cards and deck2.cards:
        a = next(deck1)
        b = next(deck2)
        if a > b:
            deck1.append(a, b)
        else:
            deck2.append(b, a)
        battle_count += 1
    if deck1.cards:
        return deck1, battle_count
    return deck2, battle_count


SAMPLE_INPUTS = [
    """\
Player 1:
9
2
6
3
1

Player 2:
5
8
4
7
10
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read_decks(sample_input):
    expected = Deck("Player 1", deque([9, 2, 6, 3, 1])), Deck("Player 2", deque([5, 8, 4, 7, 10]))
    assert read_decks(sample_input) == expected


@pytest.mark.parametrize("reverse_decks", [False, True])
def test_run_game(sample_input, reverse_decks):
    deck1, deck2 = read_decks(sample_input)
    expected_winner = deck2
    if reverse_decks:
        deck1, deck2 = deck2, deck1
    winning_deck, battle_count = run_game(deck1, deck2)
    assert battle_count == 29
    assert winning_deck is expected_winner


def test_calc_score(sample_input):
    decks = read_decks(sample_input)
    winning_deck, _ = run_game(*decks)
    assert winning_deck.calc_score() == 306
