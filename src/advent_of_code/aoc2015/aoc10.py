"""Advent of Code 2015, day 10: https://adventofcode.com/2015/day/10"""

from __future__ import annotations

from functools import cache

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            line = f.readline().rstrip()
        for _ in range(40):
            line = next_step(line)
        return len(line)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            line = f.readline().rstrip()
        for _ in range(50):
            line = next_step(line)
        return len(line)


@cache
def next_step(text: str) -> str:
    if all_same_char(text):
        return f"{len(text)}{text[0]}"

    left, right = split_on_dislike_chars(text)
    return next_step(left) + next_step(right)


def all_same_char(text: str) -> bool:
    return all(c == text[0] for c in text[1:])


def split_on_dislike_chars(text: str) -> tuple[str, str]:
    m = len(text) // 2
    left, right = text[:m], text[m:]
    while right and left[-1] == right[0]:
        left, right = left + right[:1], right[1:]
    if not right:
        left, right = text[:m], text[m:]
        while left and left[-1] == right[0]:
            left, right = left[:-1], left[-1:] + right
    return left, right


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", "11"),
        ("11", "21"),
        ("21", "1211"),
        ("1211", "111221"),
        ("111221", "312211"),
        ("112222", "2142"),
    ],
)
def test_next_step(text, expected):
    assert next_step(text) == expected
