"""Advent of Code 2015, day 11: https://adventofcode.com/2015/day/11"""
from __future__ import annotations

import re
from io import StringIO
from typing import Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[str, str]):
    def __init__(self, **kwargs):
        super().__init__(11, 2015, **kwargs)

    def solve_part_one(self) -> str:
        with self.open_input() as f:
            return next_valid_password(f.read().strip())

    def solve_part_two(self) -> str:
        part_one_solution = self.solve_part_one()
        return next_valid_password(part_one_solution)


LOWER_CASE_LETTERS = bytearray(range(ord("a"), ord("z") + 1))
FORBIDDEN_CHARS = "ilo"
VALID_SEQUENCES = [
    result
    for i in range(len(LOWER_CASE_LETTERS) - 2)
    for result in (LOWER_CASE_LETTERS[i : i + 3].decode("ASCII"),)
    if not any(c in result for c in FORBIDDEN_CHARS)
]


def increment_password(text: str) -> Iterator[str]:
    """Generate candidate passwords."""
    bs = bytearray(text, encoding="ASCII")
    if (_match := re.search(rf"[{FORBIDDEN_CHARS}]", text)) is not None:
        bs[_match.start()] += 1
        for i in range(_match.start() + 1, len(bs)):
            bs[i] = ord("a")
    while True:
        bs[-1] += 1
        for i in range(len(bs) - 1, -1, -1):
            if chr(bs[i]) in FORBIDDEN_CHARS:
                bs[i] += 1
            if bs[i] <= ord("z"):
                break
            bs[i] = ord("a")
            if i > 0:
                bs[i - 1] += 1

        yield bs.decode("ASCII")


def is_password_valid(text: str) -> bool:
    if re.search(r"(.)\1.*(.)\2", text) is None:
        return False
    return any(seq in text for seq in VALID_SEQUENCES)


def next_valid_password(password: str) -> str:
    for candidate in increment_password(password):
        if is_password_valid(candidate):
            return candidate
    raise ValueError("No valid password found.")


SAMPLE_INPUTS = [
    """\
abcdefgh
""",
    """\
ghijklmn
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, "abcdffaa"), (1, "ghjaabcc")], indirect=["sample_input"]
)
def test_next_valid_password(sample_input, expected):
    assert next_valid_password(sample_input.read().strip()) == expected


@pytest.mark.parametrize(
    ("password", "is_valid"),
    [
        ("abcd", False),
        ("AbCdEfGh", False),
        ("hijklmmn", False),
        ("abbceffg", False),
        ("abbcegjk", False),
        ("abcdffaa", True),
        ("ghjaabcc", True),
        ("ghjaaaaa", False),
    ],
)
def test_is_password_valid(password, is_valid):
    assert is_password_valid(password) is is_valid
