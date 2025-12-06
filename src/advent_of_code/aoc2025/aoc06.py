"""Advent of Code 2025, day 6: https://adventofcode.com/2025/day/6"""

from __future__ import annotations

import logging
import operator
from functools import reduce
from io import StringIO
from typing import IO, Iterable, Iterator, Optional

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            problems, operators = read_problems(fp)
        return sum(solve_problems(problems, operators))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            problems, operators = read_problems_2(fp)
        return sum(solve_problems(problems, operators))


def read_problems(reader: IO) -> tuple[list[list[int]], list[str]]:
    problem_lines: list[Iterable[int]] = []
    for line in reader:
        if line.lstrip()[0].isnumeric():
            problem_lines.append((int(n) for n in line.split()))
        else:
            operators = line.split()
    return list(zip(*problem_lines)), operators


def read_problems_2(reader: IO) -> tuple[list[list[int]], list[str]]:
    problem_lines: list[str] = []
    for line in reader:
        if line.lstrip()[0].isnumeric():
            problem_lines.append(line)
        else:
            operators = line.split()

    problems: list[list[int]] = []
    values: list[int] = []
    for col in zip(*problem_lines):
        value_str = "".join(col).strip()
        if value_str:
            values.append(int(value_str))
        else:
            problems.append(values)
            values = []
    problems.append(values)

    return problems, operators


def solve_problems(problems: list[list[int]], operators: list[str]) -> Iterator[int]:
    for prob, op in zip(problems, operators):
        match op:
            case "*":
                yield reduce(operator.mul, prob)
            case "+":
                yield reduce(operator.add, prob)
            case _:
                raise ValueError(f"Unexpected operator: '{op}'.")


SAMPLE_INPUTS = [
    """\
123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 4277556


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 3263827


def test_part_two_results(sample_input: IO) -> None:
    problems, operators = read_problems_2(sample_input)
    assert operators == ["*", "+", "*", "+"]
    assert problems == [
        [1, 24, 356], [369, 248, 8], [32, 581, 175], [623, 431, 4]
    ]
    assert list(solve_problems(problems, operators)) == [8544, 625, 3253600, 1058]
