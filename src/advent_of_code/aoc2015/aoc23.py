"""Advent of Code 2015, day 23: https://adventofcode.com/2015/day/23"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional, TypeAlias

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(23, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = [Instruction.parse(line) for line in f]
        comp = Computer()
        comp.run(program)
        return comp.registers["b"]

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = [Instruction.parse(line) for line in f]
        comp = Computer()
        comp.registers["a"] = 1
        comp.run(program)
        return comp.registers["b"]


@dataclass(frozen=True)
class Instruction:
    instruction: str
    register: Optional[str]
    offset: Optional[int]

    @classmethod
    def parse(cls, line: str) -> Instruction:
        pattern = r"([a-z]{3}) (a|b)?,?(\s?[+-][0-9]+)?"
        match = re.match(pattern, line)
        inst, reg, offset = match.groups()
        if offset is not None:
            offset = int(offset.strip())
        return cls(inst, reg, offset)


Program: TypeAlias = list[Instruction]


@dataclass
class Computer:
    registers: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0})
    pointer: int = 0

    def run(self, program: Program):
        while 0 <= self.pointer < len(program):
            self.execute(program[self.pointer])

    def execute(self, instruction: Instruction):
        log.debug("Executing %s on %s", instruction, self)
        match instruction.instruction:
            case "hlf":  # halve
                self.registers[instruction.register] //= 2
                self.pointer += 1
            case "tpl":  # triple
                self.registers[instruction.register] *= 3
                self.pointer += 1
            case "inc":  # increment
                self.registers[instruction.register] += 1
                self.pointer += 1
            case "jmp":  # jump
                self.pointer += instruction.offset
            case "jie":  # jump if even
                self.pointer += (
                    1 if self.registers[instruction.register] % 2 else instruction.offset
                )
            case "jio":  # jump if one
                self.pointer += (
                    instruction.offset if self.registers[instruction.register] == 1 else 1
                )


SAMPLE_INPUTS = [
    """\
inc a
jio a, +2
tpl a
inc a
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, {"a": 2, "b": 0})], indirect=["sample_input"]
)
def test_run_program(sample_input, expected):
    comp = Computer()
    program = [Instruction.parse(line) for line in sample_input]
    comp.run(program)
    assert comp.registers == expected
