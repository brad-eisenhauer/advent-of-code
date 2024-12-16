"""Advent of Code 2024, day 3: https://adventofcode.com/2024/day/3"""

from __future__ import annotations

import re
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            text = reader.read()
        return sum(proc_mul_ops(text))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            text = reader.read()
        return sum(proc_mul_ops(text, conditionals=True))


REGEX = r"(?P<op>mul)\((?P<left>\d+),(?P<right>\d+)\)"
REGEX_WITH_CONDITIONS = r"(?P<op>mul|do|don't)\((?:(?P<left>\d+),(?P<right>\d+))?\)"


def proc_mul_ops(text: str, conditionals: bool = False) -> Iterator[int]:
    pattern = REGEX_WITH_CONDITIONS if conditionals else REGEX
    flag = True
    for match in re.finditer(pattern, text):
        match match.groupdict()["op"], flag:
            case "mul", True:
                yield int(match.groupdict()["left"]) * int(match.groupdict()["right"])
            case "do", _:
                flag = True
            case "don't", _:
                flag = False


SAMPLE_INPUTS = [
    """\
xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))
""",
    """\
xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 161


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 48
