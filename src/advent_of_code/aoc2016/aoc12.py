"""Advent of Code 2016, day 12: https://adventofcode.com/2016/day/12"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import StringIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2016, **kwargs)

    def solve_part_one(self) -> int:
        interpreter = AssembunnyInterpreter()
        with self.open_input() as fp:
            interpreter.run(fp.readlines())
        return interpreter.registers["a"]

    def solve_part_two(self) -> int:
        interpreter = AssembunnyInterpreter()
        interpreter.registers["c"] = 1
        with self.open_input() as fp:
            interpreter.run(fp.readlines())
        return interpreter.registers["a"]


@dataclass
class AssembunnyInterpreter:
    registers: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "c": 0, "d": 0})
    pointer: int = 0
    program: list[str] = field(default_factory=list)

    def run(self, program: list[str]):
        self.program = program
        self.pointer = 0
        while self.pointer < len(self.program):
            self.execute_instruction()

    def execute_instruction(self):
        pattern = r"(?P<inst>[a-z]{3}) (?P<arg1>[a-d]|-?[0-9]+)(?: (?P<arg2>[a-d]|-?[0-9]+))?"
        match = re.match(pattern, self.program[self.pointer])
        match match.groupdict()["inst"]:
            case "cpy":
                value = match.groupdict()["arg1"]
                value = int(value) if value.isnumeric() else self.registers[value]
                dest = match.groupdict()["arg2"]
                self.registers[dest] = value
                self.pointer += 1
            case "inc":
                register = match.groupdict()["arg1"]
                self.registers[register] += 1
                self.pointer += 1
            case "dec":
                register = match.groupdict()["arg1"]
                self.registers[register] -= 1
                self.pointer += 1
            case "jnz":
                value = match.groupdict()["arg1"]
                value = int(value) if value.isnumeric() else self.registers[value]
                jump = int(match.groupdict()["arg2"])
                if value == 0:
                    self.pointer += 1
                else:
                    self.pointer += jump


SAMPLE_INPUTS = [
    """\
cpy 41 a
inc a
inc a
dec a
jnz a 2
dec a
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_run(sample_input):
    interpreter = AssembunnyInterpreter()
    interpreter.run(sample_input.readlines())
    assert interpreter.registers["a"] == 42
