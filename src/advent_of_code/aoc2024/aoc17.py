"""Advent of Code 2024, day 17: https://adventofcode.com/2024/day/17"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[str, int | None]):
    def __init__(self, **kwargs):
        super().__init__(17, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> str:
        with input_file or self.open_input() as reader:
            computer = ChronospatialComputer.read(reader)
        return ",".join(str(n) for n in computer.run())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int | None:
        with input_file or self.open_input() as reader:
            computer = ChronospatialComputer.read(reader)
        candidates: deque[tuple[int, int]] = deque([(0, 0)])
        final_answers: list[int] = []
        while candidates:
            prefix, match_count = candidates.pop()
            if match_count == len(computer.program):
                final_answers.append(prefix)
                continue
            for a in range(8):
                value = (prefix << 3) + a
                computer.registers = {"A": value, "B": 0, "C": 0}
                computer.pointer = 0
                result = list(computer.run())
                if result == computer.program[-(match_count + 1) :]:
                    log.debug("A=%d matched last %d values.", value, len(result))
                    if len(result) == len(computer.program):
                        return value
                    candidates.append((value, len(result)))
        return None


@dataclass
class ChronospatialComputer:
    program: list[int]
    registers: dict[str, int]
    pointer: int = 0

    @classmethod
    def read(cls, reader: IO) -> ChronospatialComputer:
        registers: dict[str, int] = {}
        registers["A"] = int(re.search(r"\d+", reader.readline()).group())
        registers["B"] = int(re.search(r"\d+", reader.readline()).group())
        registers["C"] = int(re.search(r"\d+", reader.readline()).group())
        reader.readline()
        program: list[int] = [int(n) for n in reader.readline().split(": ")[1].split(",")]
        return cls(program=program, registers=registers)

    def read_literal(self) -> int:
        return self.program[self.pointer + 1]

    def read_combo(self) -> int:
        operand = self.read_literal()
        if operand < 4:
            return operand
        if operand < 7:
            register = chr(ord("A") + operand - 4)
            return self.registers[register]
        raise ValueError(f"Unexpected operand value: {operand}")

    def adv(self) -> None:
        """Opcode 0"""
        self._xdv("A")
        self.pointer += 2

    def bxl(self) -> None:
        """Opcode 1"""
        self.registers["B"] ^= self.read_literal()
        self.pointer += 2

    def bst(self) -> None:
        """Opcode 2"""
        self.registers["B"] = self.read_combo() % 8
        self.pointer += 2

    def jnz(self) -> None:
        """Opcode 3"""
        if self.registers["A"] == 0:
            self.pointer += 2
        else:
            self.pointer = self.read_literal()

    def bxc(self) -> None:
        """Opcode 4"""
        self.registers["B"] ^= self.registers["C"]
        self.pointer += 2

    def out(self) -> int:
        """Opcode 5"""
        result = self.read_combo() % 8
        self.pointer += 2
        return result

    def bdv(self) -> None:
        """Opcode 6"""
        self._xdv("B")
        self.pointer += 2

    def cdv(self) -> None:
        """Opcode 7"""
        self._xdv("C")
        self.pointer += 2

    def _xdv(self, dest_register: str) -> None:
        self.registers[dest_register] = self.registers["A"] // 2 ** self.read_combo()

    def step(self) -> int | None:
        match self.program[self.pointer]:
            case 0:
                self.adv()
            case 1:
                self.bxl()
            case 2:
                self.bst()
            case 3:
                self.jnz()
            case 4:
                self.bxc()
            case 5:
                return self.out()
            case 6:
                self.bdv()
            case 7:
                self.cdv()
            case op_code:
                raise ValueError(f"Unexpected op code: {op_code}")
        return None

    def run(self) -> Iterator[int]:
        while self.pointer < len(self.program):
            if (out := self.step()) is not None:
                yield out


SAMPLE_INPUTS = [
    """\
Register A: 729
Register B: 0
Register C: 0

Program: 0,1,5,4,3,0
""",
    """\
Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == "4,6,3,5,6,3,5,2,1,0"


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 117440
