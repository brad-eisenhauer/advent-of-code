"""Advent of Code 2015, day 5: https://adventofcode.com/2015/day/5"""
import re
from io import StringIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(5, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            result = sum(1 for line in f if is_nice(line.strip()))
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            result = sum(1 for line in f if is_nice_2(line.strip()))
        return result


def is_nice(text: str) -> bool:
    BAD_WORDS = ["ab", "cd", "pq", "xy"]
    if not any(a == b for a, b in zip(text, text[1:])):
        return False
    if any(bw in text for bw in BAD_WORDS):
        return False
    if sum(1 for c in text if c in {"a", "e", "i", "o", "u"}) < 3:
        return False
    return True


def is_nice_2(text: str) -> bool:
    rule_1 = re.compile(r"(.{2}).*(\1)")
    rule_2 = re.compile(r"(.).(\1)")
    return rule_1.search(text) is not None and rule_2.search(text) is not None


SAMPLE_INPUTS = [
    "ugknbfddgicrmopn",
    "jchzalrnumimnmhp",
    "haegwjzuvuyypxyu",
    "dvszwmarrgswjxmb",
    "qjhvhtzxzqqjkmpb",
    "xxyxx",
    "uurcxstgmygtbstg",
    "ieodomkazucvgmuy",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, True), (1, False), (2, False), (3, False)],
    indirect=["sample_input"],
)
def test_is_nice(sample_input, expected):
    assert is_nice(sample_input.readline().strip()) is expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(4, True), (5, True), (6, False), (7, False)],
    indirect=["sample_input"],
)
def test_is_nice_2(sample_input, expected):
    assert is_nice_2(sample_input.readline().strip()) is expected
