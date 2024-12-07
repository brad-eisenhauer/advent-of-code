"""Advent of Code 2024, day 7: https://adventofcode.com/2024/day/7"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from io import StringIO
from itertools import product
from typing import IO, Callable, Collection, Iterable, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            return sum(
                eq.result for line in reader if (eq := EmptyEquation.read(line)).is_solvable()
            )

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            return sum(
                eq.result
                for line in reader
                if (eq := EmptyEquation.read(line)).is_solvable(
                    (operator.add, operator.mul, concat)
                )
            )


@dataclass
class EmptyEquation:
    result: int
    operands: list[int]

    @classmethod
    def read(cls, text: str) -> EmptyEquation:
        result_str, operands_str = text.split(":")
        operands = [int(n) for n in operands_str.split()]
        return cls(result=int(result_str), operands=operands)

    def eval_operands_with(self, operators: Iterable[Callable[[int, int], int]]) -> int:
        result = self.operands[0]
        for op, operand in zip(operators, self.operands[1:]):
            result = op(result, operand)
        return result

    def is_solvable(
        self, operators: Collection[Callable[[int, int], int]] = (operator.add, operator.mul)
    ) -> bool:

        def _is_solvable_iter(operands = self.operands) -> bool:
            if len(operands) == 1:
                return operands[0] == self.result
            if operands[0] > self.result:
                return False
            for op in operators:
                new_operands = [op(*operands[:2])] + operands[2:]
                if _is_solvable_iter(new_operands):
                    return True
            return False

        return _is_solvable_iter()


def concat(left: int, right: int) -> int:
    shift = math.floor(math.log10(right)) + 1
    return left * 10**shift + right


SAMPLE_INPUTS = [
    """\
190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_read(sample_input: IO) -> None:
    equations = [EmptyEquation.read(line) for line in sample_input]
    assert equations == [
        EmptyEquation(190, [10, 19]),
        EmptyEquation(3267, [81, 40, 27]),
        EmptyEquation(83, [17, 5]),
        EmptyEquation(156, [15, 6]),
        EmptyEquation(7290, [6, 8, 6, 15]),
        EmptyEquation(161011, [16, 10, 13]),
        EmptyEquation(192, [17, 8, 14]),
        EmptyEquation(21037, [9, 7, 18, 13]),
        EmptyEquation(292, [11, 6, 16, 20]),
    ]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (1, 1, 11),
        (123, 45, 12345),
        (34, 9, 349),
        (34, 10, 3410),
    ],
)
def test_concat(left, right, expected) -> None:
    assert concat(left, right) == expected


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 3749


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 11387
