"""Advent of Code 2020, day 2: https://adventofcode.com/2020/day/2"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Protocol, Type, TypeVar

import pytest

from advent_of_code.base import Solution

PolicyT = TypeVar("PolicyT")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            result = sum(1 for line in f if PasswordEntry.parse(line).is_valid())
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            result = sum(1 for line in f if PasswordEntry.parse(line, Policy2).is_valid())
        return result


class Policy(Protocol):
    @classmethod
    def parse(cls: Type[PolicyT], text: str) -> PolicyT:
        ...

    def validate(self, password: str) -> bool:
        ...


@dataclass
class Policy1(Policy):
    char: str
    count: range

    @classmethod
    def parse(cls, text: str) -> Policy1:
        count_range, char = text.split()
        min_count, max_count = map(int, count_range.split("-"))
        return cls(char, range(min_count, max_count + 1))

    def validate(self, password: str) -> bool:
        char_counts = sum(1 for c in password if c == self.char)
        return char_counts in self.count


@dataclass
class Policy2(Policy):
    char: str
    positions: list[int]

    @classmethod
    def parse(cls, text: str) -> Policy2:
        position_str, char = text.split()
        positions = [int(n) for n in position_str.split("-")]
        return cls(char, positions)

    def validate(self, password: str) -> bool:
        left, right = self.positions
        return (password[left - 1] == self.char) != (password[right - 1] == self.char)


@dataclass
class PasswordEntry:
    password: str
    policy: Policy

    @classmethod
    def parse(cls, text: str, policy_cls: Type[Policy] = Policy1) -> PasswordEntry:
        policy_str, password = text.split(":")
        return cls(password.strip(), policy_cls.parse(policy_str))

    def is_valid(self) -> bool:
        return self.policy.validate(self.password)


SAMPLE_INPUT = """\
1-3 a: abcde
1-3 b: cdefg
2-9 c: ccccccccc
"""


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_parse_passwords(sample_input):
    expected = [
        PasswordEntry("abcde", Policy1("a", range(1, 4))),
        PasswordEntry("cdefg", Policy1("b", range(1, 4))),
        PasswordEntry("ccccccccc", Policy1("c", range(2, 10))),
    ]
    result = [PasswordEntry.parse(line) for line in sample_input]
    assert result == expected


@pytest.mark.parametrize(
    ("policy_cls", "expected"),
    [
        (Policy1, [True, False, True]),
        (Policy2, [True, False, False]),
    ],
)
def test_validate_passwords(sample_input, policy_cls, expected):
    result = [PasswordEntry.parse(line, policy_cls).is_valid() for line in sample_input]
    assert result == expected
