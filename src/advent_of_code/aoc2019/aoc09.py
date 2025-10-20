"""Advent of Code 2019, Day 9: https://adventofcode.com/2019/day/9"""

from math import floor, log10

import pytest

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        machine = IntcodeMachine(program, iter((1,)))
        result = list(machine.run())
        assert len(result) == 1
        return result[0]

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        machine = IntcodeMachine(program, iter((2,)))
        result = list(machine.run())
        assert len(result) == 1
        return result[0]


def test_quine():
    quine = [109, 1, 204, -1, 1001, 100, 1, 100, 1008, 100, 16, 101, 1006, 101, 0, 99]
    machine = IntcodeMachine(quine.copy())
    result = list(machine.run())
    assert result == quine


def test_large_number():
    program = [1102, 34915192, 34915192, 7, 4, 7, 99, 0]
    machine = IntcodeMachine(program)
    result = next(machine.run())
    assert floor(log10(result)) + 1 == 16


def test_large_number_2():
    program = [104, 1125899906842624, 99]
    machine = IntcodeMachine(program.copy())
    result = next(machine.run())
    assert result == program[1]


def test_read_past_initial_buffer():
    program = [4, 100, 99]
    machine = IntcodeMachine(program)
    [result] = list(machine.run())
    assert result == 0


def test_write_past_initial_buffer():
    program = [3, 100, 4, 100, 99]
    machine = IntcodeMachine(program, input_stream=iter((42,)))
    [result] = list(machine.run())
    assert result == 42


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ([3, 100, 9, 1, 204, 99, 204, 0, 99], [0, 42]),
        ([3, 100, 109, 1, 204, 99, 204, 0, 99], [42, 100]),
    ],
)
def test_relative_mode(program, expected):
    machine = IntcodeMachine(program, input_stream=iter((42,)))
    result = list(machine.run())
    assert result == expected


def test_relative_mode_input():
    program = [203, 2]
    machine = IntcodeMachine(program, input_stream=iter((99,)))
    result = list(machine.run())
    assert result == []
