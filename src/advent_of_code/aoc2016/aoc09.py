"""Advent of Code 2016, day 9: https://adventofcode.com/2016/day/9"""
from __future__ import annotations

import re

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            return decompressed_len(fp.read().strip())

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            return decompressed_len(fp.read().strip(), recursive=True)


def decompressed_len(text: str, recursive: bool = False) -> int:
    pattern = r"(?P<prefix>.*?)(\((?P<char_count>\d+)x(?P<rep_count>\d+)\))"
    result = 0
    while (match := re.match(pattern, text)) is not None:
        result += len(match.groupdict()["prefix"])
        text = text[match.end() :]
        char_count = int(match.groupdict()["char_count"])
        rep_count = int(match.groupdict()["rep_count"])
        chunk = text[:char_count]
        result += rep_count * (decompressed_len(chunk, recursive) if recursive else len(chunk))
        text = text[char_count:]
    result += len(text)
    return result


@pytest.mark.parametrize(
    ("compressed_text", "recursive", "expected"),
    [
        ("ADVENT", False, 6),
        ("A(1x5)BC", False, 7),
        ("(3x3)XYZ", False, 9),
        ("A(2x2)BCD(2x2)EFG", False, 11),
        ("(6x1)(1x3)A", False, 6),
        ("X(8x2)(3x3)ABCY", False, 18),
        ("(3x3)XYZ", True, 9),
        ("X(8x2)(3x3)ABCY", True, 20),
        ("(27x12)(20x12)(13x14)(7x10)(1x12)A", True, 241920),
        ("(25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN", True, 445),
    ],
)
def test_decompress(compressed_text, recursive, expected):
    assert decompressed_len(compressed_text, recursive) == expected
