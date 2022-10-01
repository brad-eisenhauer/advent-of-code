""" Advent of Code 2019, Day 5: https://adventofcode.com/2019/day/5 """
from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(5, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            machine = IntcodeMachine(IntcodeMachine.read_buffer(f), iter((1,)))
        result = -1
        for result in machine.run():
            ...
        return result


def test_read_write():
    machine = IntcodeMachine([3, 0, 4, 0, 99], iter((42,)))
    result = list(machine.run())
    assert result == [42]


def test_param_modes():
    machine = IntcodeMachine([1002, 4, 3, 4, 33])
    result = list(machine.run())
    assert result == []


def test_program_with_negative():
    machine = IntcodeMachine([1101, 100, -1, 4, 0])
    result = list(machine.run())
    assert result == []
