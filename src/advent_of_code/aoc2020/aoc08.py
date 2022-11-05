"""Advent of Code 2020, day 8: https://adventofcode.com/2020/day/8"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from typing import Sequence, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = Program.read(f)
        return program.run()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = Program.read(f)
        _, result = fix_program(program)
        return result


class LoopError(RuntimeError):
    ...


@dataclass
class Instruction:
    instruction: str
    value: int
    call_count: int = 0


@dataclass
class Program:
    instructions: Sequence[Instruction]
    acc: int = 0
    pointer: int = 0

    @classmethod
    def read(cls, f: TextIO) -> Program:
        instructions = []
        for line in f.readlines():
            op, v = line.split()
            instructions.append(Instruction(op, int(v)))
        return cls(instructions)

    def run(self, raise_on_loop: bool = False) -> int:
        while True:
            if self.pointer >= len(self.instructions):
                return self.acc
            try:
                self.step()
            except LoopError:
                if raise_on_loop:
                    raise
                return self.acc

    def step(self):
        instruction = self.instructions[self.pointer]
        if instruction.call_count > 0:
            raise LoopError()
        instruction.call_count += 1
        match instruction.instruction:
            case "nop":
                ...
            case "acc":
                self.acc += instruction.value
            case "jmp":
                self.pointer += instruction.value
                return
        self.pointer += 1


def fix_program(program: Program) -> tuple[Program, int]:
    while True:
        next_instruction = program.instructions[program.pointer].instruction
        if next_instruction in ["jmp", "nop"]:
            dup_program = deepcopy(program)
            dup_program.instructions[dup_program.pointer].instruction = (
                "nop" if next_instruction == "jmp" else "jmp"
            )
            try:
                result = dup_program.run(raise_on_loop=True)
                return dup_program, result
            except LoopError:
                ...
        program.step()


SAMPLE_INPUT = """\
nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_read_program(sample_input):
    result = Program.read(sample_input)
    expected = Program(
        [
            Instruction("nop", 0),
            Instruction("acc", 1),
            Instruction("jmp", 4),
            Instruction("acc", 3),
            Instruction("jmp", -3),
            Instruction("acc", -99),
            Instruction("acc", 1),
            Instruction("jmp", -4),
            Instruction("acc", 6),
        ]
    )
    assert result == expected


def test_run_program(sample_input):
    program = Program.read(sample_input)
    result = program.run()
    assert result == 5


def test_fix_program(sample_input):
    program = Program.read(sample_input)
    fixed_program, result = fix_program(program)
    assert result == 8
