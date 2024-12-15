"""Advent of Code 2017, day 8: https://adventofcode.com/2017/day/8"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import IO, ClassVar, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        registers = defaultdict(int)
        registers["_max"] = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                Instruction.from_str(line).proc(registers)
        return max(v for k, v in registers.items() if k != "_max")

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        registers = defaultdict(int)
        registers["_max"] = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                Instruction.from_str(line).proc(registers)
        return registers["_max"]


@dataclass
class Instruction:
    register: str
    op: str
    amount: int
    condition: str

    # fmt: off
    REGEX: ClassVar[str] = (
        r"(?P<register>[a-z]+) "  # register name, followed by a space
        r"(?P<op>inc|dec) "       # operation, followed by a space
        r"(?P<amount>-*[0-9]+) "  # amount, followed by a space
        r"if "                    # literal 'if'
        r"(?P<condition>[a-z]+ [<>=!]{1,2} -*[0-9]+)"  # condition
    )
    # fmt: on

    @classmethod
    def from_str(cls, text: str) -> Instruction:
        match_ = re.match(cls.REGEX, text)
        if match_ is None:
            raise ValueError(f"Could not parse input: '{text}'.")
        return cls(
            register=match_.groupdict()["register"],
            op=match_.groupdict()["op"],
            amount=int(match_.groupdict()["amount"]),
            condition=match_.groupdict()["condition"],
        )

    def eval(self, registers: dict[str, int]) -> bool:
        register = self.condition.split()[0]
        register_value = registers[register]
        condition = " ".join([str(register_value), *self.condition.split()[1:]])
        return eval(condition)

    def proc(self, registers: defaultdict[str, int]) -> None:
        if self.eval(registers):
            increment = self.amount if self.op == "inc" else -self.amount
            registers[self.register] += increment
            registers["_max"] = max(registers["_max"], registers[self.register])


SAMPLE_INPUTS = [
    """\
b inc 5 if a > 1
a inc 1 if b < 5
c dec -10 if a >= 1
c inc -20 if c == 10
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
    assert solution.solve_part_one(sample_input) == 1


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 10
