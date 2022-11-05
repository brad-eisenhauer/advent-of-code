"""Advent of Code 2019, day 25: https://adventofcode.com/2019/day/25"""
import sys
from itertools import takewhile
from typing import Iterator

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(25, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        machine = InteractiveAsciiMachine(program)
        machine.run()


class InteractiveAsciiMachine:
    def __init__(self, program: list[int]):
        self.controller = IntcodeMachine(program, input_stream=self.input())

    def input(self) -> Iterator[int]:
        while True:
            sys.stdout.flush()
            command = sys.stdin.readline()
            yield from command.encode("ASCII")

    def run(self):
        output_stream = self.controller.run()
        try:
            while self.controller.pointer is not None:
                output = takewhile(lambda char: char != 10, output_stream)
                sys.stdout.write(bytes(output).decode("ASCII") + "\n")
        except StopIteration:
            pass
