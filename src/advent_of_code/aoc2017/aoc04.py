"""Advent of Code 2017, day 4: https://adventofcode.com/2017/day/4"""
from __future__ import annotations

from collections import Counter
from io import StringIO
from typing import IO, Hashable, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            return sum(1 for line in fp if is_passphrase_valid(line))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            return sum(1 for line in fp if is_passphrase_valid_2(line))


def is_passphrase_valid(text: str) -> bool:
    words = text.split()
    return len(words) == len(set(words))


def is_passphrase_valid_2(text: str) -> bool:
    if not is_passphrase_valid(text):
        return False

    def _canonical_word(word: str) -> Hashable:
        c = Counter(word)
        return tuple(sorted(c.items()))

    words = text.split()
    canonical_words = set(_canonical_word(w) for w in words)
    return len(words) == len(canonical_words)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("abcde fghij", True),
        ("abcde xyz ecdab", False),
        ("a ab abc abd abf abj", True),
        ("iiii oiii ooii oooi oooo", True),
        ("oiii ioii iioi iiio", False),
    ],
)
def test_is_passphrase_valid_2(phrase: str, expected: bool) -> None:
    assert is_passphrase_valid_2(phrase) == expected
