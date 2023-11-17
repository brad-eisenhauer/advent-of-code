"""Advent of Code 2016, day 7: https://adventofcode.com/2016/day/7"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            return sum(1 for address in fp if IPv7Address(address).supports_tls())

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            return sum(1 for address in fp if IPv7Address(address).supports_ssl())


@dataclass
class IPv7Address:
    address: str

    def supports_tls(self) -> bool:
        if any(self.contains_abba(seq) for seq in self.hypernet_sequences()):
            return False
        return any(self.contains_abba(seq) for seq in self.supernet_sequences())

    def supports_ssl(self) -> bool:
        def find_abas(seq: str) -> Iterator[str]:
            pattern = r"([a-z])([a-z])\1"
            while seq and (match := re.search(pattern, seq)) is not None:
                aba = match.group(0)
                if aba[0] != aba[1]:
                    yield match.group(0)
                seq = seq[match.start() + 1 :]

        supernet_abas = (aba for sup_seq in self.supernet_sequences() for aba in find_abas(sup_seq))
        for aba in supernet_abas:
            bab = f"{aba[1]}{aba[0]}{aba[1]}"
            if any(bab in seq for seq in self.hypernet_sequences()):
                return True
        return False

    def hypernet_sequences(self) -> Iterator[str]:
        pattern = r"\[([a-z]+)\]"
        remainder = self.address
        while True:
            if (match := re.search(pattern, remainder)) is None:
                break
            yield match.group(1)
            remainder = remainder[match.end() :]

    def supernet_sequences(self) -> Iterator[str]:
        pattern = r"^([^\[\]]+)(?:\[[^\[\]]+\])?"
        remainder = self.address
        while True:
            if (match := re.match(pattern, remainder)) is None:
                break
            yield match.group(1)
            remainder = remainder[match.end() :]

    @staticmethod
    def contains_abba(sequence: str) -> bool:
        pattern = r"([a-z])([a-z])\2\1"
        if (match := re.search(pattern, sequence)) is None:
            return False
        abba = match.groups(0)
        return abba[0] != abba[1]


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("abba[mnop]qrst", ["mnop"]),
        ("abcd[bddb]xyyx", ["bddb"]),
        ("aaaa[qwer]tyui", ["qwer"]),
        ("ioxxoj[asdfgh]zxcvbn", ["asdfgh"]),
    ],
)
def test_ipv7_hypernet_sequences(address, expected):
    assert list(IPv7Address(address).hypernet_sequences()) == expected


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("abba[mnop]qrst", True),
        ("abcd[bddb]xyyx", False),
        ("aaaa[qwer]tyui", False),
        ("ioxxoj[asdfgh]zxcvbn", True),
    ],
)
def test_ipv7_supports_tls(address, expected):
    assert IPv7Address(address).supports_tls() is expected


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("aba[bab]xyz", True),
        ("xyx[xyx]xyx", False),
        ("aaa[kek]eke", True),
        ("zazbz[bzb]cdb", True),
    ],
)
def test_ipv7_supports_ssl(address, expected):
    assert IPv7Address(address).supports_ssl() is expected
