"""Advent of Code 2024, day 5: https://adventofcode.com/2024/day/5"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(5, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            rules = list(read_ordering_rules(reader))
            updates = list(read_page_orderings(reader))
        result = 0
        for update in updates:
            if all(rule.validate(update) for rule in rules):
                result += update[len(update) // 2]
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            rules = list(read_ordering_rules(reader))
            updates = list(read_page_orderings(reader))
        result = 0
        for update in updates:
            if all(rule.validate(update) for rule in rules):
                continue
            fix_ordering(update, rules)
            result += update[len(update) // 2]
        return result


@dataclass
class OrderingRule:
    predecessor: int
    successor: int

    @classmethod
    def parse(cls, text: str) -> OrderingRule:
        pred, succ = (int(n) for n in text.split("|"))
        return OrderingRule(pred, succ)

    def validate(self, page_ordering: list[int]) -> bool:
        try:
            pred_index = page_ordering.index(self.predecessor)
            succ_index = page_ordering.index(self.successor)
        except ValueError:
            return True
        return pred_index < succ_index

    def reorder(self, page_ordering: list[int]) -> None:
        pred_index = page_ordering.index(self.predecessor)
        page_ordering.remove(self.successor)
        page_ordering.insert(pred_index, self.successor)


def read_ordering_rules(reader: IO) -> Iterator[OrderingRule]:
    while line := reader.readline().strip():
        yield OrderingRule.parse(line)


def read_page_orderings(reader: IO) -> Iterator[list[int]]:
    while line := reader.readline().strip():
        yield [int(n) for n in line.split(",")]


def fix_ordering(page_ordering: list[int], rules: list[OrderingRule]) -> None:
    _rules = rules
    while True:
        for i, rule in enumerate(_rules):
            if not rule.validate(page_ordering):
                rule.reorder(page_ordering)
                _rules = _rules[i + 1 :] + _rules[: i + 1]
                break
        else:
            return


SAMPLE_INPUTS = [
    """\
47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 143


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 123
