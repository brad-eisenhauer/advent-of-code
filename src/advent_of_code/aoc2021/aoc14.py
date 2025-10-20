"""Advent of Code 2021, Day 14: https://adventofcode.com/2021/day/14"""

from __future__ import annotations

from collections import Counter
from functools import cache
from io import StringIO
from typing import Iterable, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            template, formula = read_input(f)
        return formula.calc_element_count_range(template, 10)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            template, formula = read_input(f)
        return formula.calc_element_count_range(template, 40)


def read_input(fp: TextIO) -> tuple[str, Formula]:
    template = next(fp).strip()
    _ = next(fp)  # throw away empty line
    formula = Formula.read(fp)
    return template, formula


class Formula:
    def __init__(self, rules: Iterable[str]):
        self.rules = {k: v for k, _, v in (line.strip().split() for line in rules)}

    @classmethod
    def read(cls, fp: TextIO) -> Formula:
        return cls(fp)

    @cache
    def count_elements(self, template: str, steps: int) -> Counter:
        if steps == 0:
            return Counter(template)

        if len(template) > 2:
            # split string and count each part separately
            mid_idx = len(template) // 2
            left = template[: mid_idx + 1]
            right = template[mid_idx:]
            result = self.count_elements(left, steps) + self.count_elements(right, steps)
            # middle character is included in both counts
            result[template[mid_idx]] -= 1
            return result

        new_char = self.rules[template]
        result = self.count_elements(template[0] + new_char, steps - 1) + self.count_elements(
            new_char + template[1], steps - 1
        )
        result[new_char] -= 1
        return result

    def calc_element_count_range(self, template: str, steps: int) -> int:
        counts = self.count_elements(template, steps)
        min_count = min(counts.values())
        max_count = max(counts.values())
        return max_count - min_count


SAMPLE_INPUT = """\
NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


@pytest.mark.parametrize(("steps", "expected"), [(10, 1588), (40, 2188189693529)])
def test_calc_element_count_range(sample_input, steps, expected):
    template, formula = read_input(sample_input)
    result = formula.calc_element_count_range(template, steps)
    assert result == expected
