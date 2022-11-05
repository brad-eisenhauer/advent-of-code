"""Advent of Code 2019, day 21: https://adventofcode.com/2019/day/21"""
import logging

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2019, **kwargs)

    def run_springscript(self, script: list[str]) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        script_bytes = ("\n".join(script) + "\n").encode("ASCII")
        machine = IntcodeMachine(program, iter(script_bytes))
        result = list(machine.run())
        if result[-1] == 10:
            log.debug(bytes(result).decode("ASCII"))
            raise SpringScriptError()
        return result[-1]

    def solve_part_one(self) -> int:
        script = [
            "OR D J",  # Jump if we can.
            "OR A T",  # Unless...
            "AND B T",
            "AND C T",
            "NOT T T",
            "AND T J",  # A, B, and C are all True.
            "WALK",
        ]
        return self.run_springscript(script)

    def solve_part_two(self) -> int:
        script = [
            "OR D J",  # Jump if we can.
            "OR A T",  # Unless...
            "AND B T",
            "AND C T",
            "NOT T T",
            "AND T J",  # A, B, and C are all True.
            "NOT A T",  # (Reset T -> False)
            "AND A T",
            "OR E T",  # Or...
            "OR H T",
            "AND T J",  # E and H are both False.
            "RUN",
        ]
        return self.run_springscript(script)


class SpringScriptError(Exception):
    ...
