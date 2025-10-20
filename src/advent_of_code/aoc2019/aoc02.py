"""Advent of Code 2019, Day 02: https://adventofcode.com/2019/day/2"""

from copy import deepcopy
from io import StringIO
from itertools import product
from typing import Iterator, TextIO

import pytest

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(2, 2019, **kwargs)

    def get_intcode_machine(self) -> IntcodeMachine:
        with self.open_input() as f:
            buffer = IntcodeMachine.read_buffer(f)
        return IntcodeMachine(buffer)

    def solve_part_one(self) -> int:
        machine = self.get_intcode_machine()
        machine.buffer[1:3] = [12, 2]
        _ = list(machine.run())
        return machine.buffer[0]

    def solve_part_two(self) -> int:
        machine = self.get_intcode_machine()
        for noun, verb in product(range(100), range(100)):
            this_machine = deepcopy(machine)
            this_machine.buffer[1:3] = noun, verb
            _ = list(this_machine.run())
            if this_machine.buffer[0] == 19690720:
                return 100 * noun + verb


SAMPLE_INPUT = """\
1,9,10,3,2,3,11,0,99,30,40,50
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_read_intcode_buffer(sample_input):
    assert IntcodeMachine.read_buffer(sample_input) == [1, 9, 10, 3, 2, 3, 11, 0, 99, 30, 40, 50]


@pytest.mark.parametrize(
    ("pointer", "expected_result", "expected_buffer"),
    (
        (0, 4, [1, 9, 10, 70, 2, 3, 11, 0, 99, 30, 40, 50]),
        (4, 8, [150, 9, 10, 3, 2, 3, 11, 0, 99, 30, 40, 50]),
    ),
)
def test_process_opcode(sample_input, pointer, expected_result, expected_buffer):
    machine = IntcodeMachine(IntcodeMachine.read_buffer(sample_input), pointer=pointer)
    machine.step()
    assert machine.pointer == expected_result
    assert machine.buffer == expected_buffer


@pytest.mark.parametrize(
    ("buffer", "expected_buffer"),
    (
        ([1, 0, 0, 0, 99], [2, 0, 0, 0, 99]),
        ([2, 3, 0, 3, 99], [2, 3, 0, 6, 99]),
        ([2, 4, 4, 5, 99, 0], [2, 4, 4, 5, 99, 9801]),
        ([1, 1, 1, 4, 99, 5, 6, 0, 99], [30, 1, 1, 4, 2, 5, 6, 0, 99]),
    ),
)
def test_run_intcode(buffer, expected_buffer):
    machine = IntcodeMachine(buffer)
    _ = list(machine.run())
    assert machine.buffer == expected_buffer
