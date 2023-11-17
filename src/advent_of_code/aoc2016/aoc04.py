"""Advent of Code 2016, day 4: https://adventofcode.com/2016/day/4"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from io import StringIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(4, 2016, **kwargs)

    def solve_part_one(self) -> int:
        result = 0
        with self.open_input() as fp:
            for line in fp:
                room = Room.from_str(line)
                if room.is_valid():
                    result += room.sector_id
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            for line in fp:
                room = Room.from_str(line)
                if room.is_valid() and "object" in room.name:
                    return room.sector_id


@dataclass
class Room:
    encrypted_name: str
    sector_id: int
    checksum: str

    @classmethod
    def from_str(cls, description: str) -> Room:
        pattern = r"(?P<encrypted_name>[a-z]+(?:-[a-z]+)*)-(?P<sector_id>[0-9]+)\[(?P<checksum>[a-z]{5})\]"
        match = re.match(pattern, description)
        return cls(
            encrypted_name=match.groupdict()["encrypted_name"],
            sector_id=int(match.groupdict()["sector_id"]),
            checksum=match.groupdict()["checksum"],
        )

    def is_valid(self) -> bool:
        char_counts = Counter(self.encrypted_name)
        checksum = "".join(
            char
            for item in sorted(char_counts.items(), key=lambda c: (-c[1], ord(c[0])))
            if (char := item[0]) != "-"
        )[:5]
        return checksum == self.checksum

    @property
    def name(self) -> str:
        return "".join(" " if c == "-" else self._decrypt(c) for c in self.encrypted_name)

    def _decrypt(self, char: str) -> str:
        return chr((ord(char) - ord("a") + self.sector_id) % 26 + ord("a"))


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("aaaaa-bbb-z-y-x-123[abxyz]", Room("aaaaa-bbb-z-y-x", 123, "abxyz")),
        ("a-b-c-d-e-f-g-h-987[abcde]", Room("a-b-c-d-e-f-g-h", 987, "abcde")),
        ("not-a-real-room-404[oarel]", Room("not-a-real-room", 404, "oarel")),
        ("totally-real-room-200[decoy]", Room("totally-real-room", 200, "decoy")),
    ],
)
def test_room_from_str(description, expected):
    assert Room.from_str(description) == expected


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("aaaaa-bbb-z-y-x-123[abxyz]", True),
        ("a-b-c-d-e-f-g-h-987[abcde]", True),
        ("not-a-real-room-404[oarel]", True),
        ("totally-real-room-200[decoy]", False),
    ],
)
def test_room_is_valid(description, expected):
    assert Room.from_str(description).is_valid() is expected
