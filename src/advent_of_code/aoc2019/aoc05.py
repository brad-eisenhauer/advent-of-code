""" Advent of Code 2019, Day 5: https://adventofcode.com/2019/day/5 """
import pytest

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

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            machine = IntcodeMachine(IntcodeMachine.read_buffer(f), iter((5,)))
        [result] = list(machine.run())
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


@pytest.mark.parametrize(
    ("program", "input", "expected"),
    [
        # Test input == 8
        ([3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8], 8, 1),
        ([3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8], 1, 0),
        ([3, 3, 1108, -1, 8, 3, 4, 3, 99], 8, 1),
        ([3, 3, 1108, -1, 8, 3, 4, 3, 99], 1, 0),
        # Test input < 8
        ([3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8], 8, 0),
        ([3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8], 1, 1),
        ([3, 3, 1107, -1, 8, 3, 4, 3, 99], 8, 0),
        ([3, 3, 1107, -1, 8, 3, 4, 3, 99], 1, 1),
    ],
)
def test_comparisons(program, input, expected):
    machine = IntcodeMachine(program, iter((input,)))
    [result] = list(machine.run())
    assert result == expected


@pytest.mark.parametrize(
    ("program", "input", "expected"),
    [
        ([3, 12, 6, 12, 15, 1, 13, 14, 13, 4, 13, 99, -1, 0, 1, 9], 0, 0),
        ([3, 12, 6, 12, 15, 1, 13, 14, 13, 4, 13, 99, -1, 0, 1, 9], 1, 1),
        ([3, 12, 6, 12, 15, 1, 13, 14, 13, 4, 13, 99, -1, 0, 1, 9], 42, 1),
        ([3, 3, 1105, -1, 9, 1101, 0, 0, 12, 4, 12, 99, 1], 0, 0),
        ([3, 3, 1105, -1, 9, 1101, 0, 0, 12, 4, 12, 99, 1], 1, 1),
        ([3, 3, 1105, -1, 9, 1101, 0, 0, 12, 4, 12, 99, 1], 42, 1),
    ],
)
def test_jumps(program, input, expected):
    machine = IntcodeMachine(program, iter((input,)))
    [result] = list(machine.run())
    assert result == expected


@pytest.mark.parametrize(
    ("input", "expected"),
    [(1, 999), (8, 1000), (9, 1001)],
)
def test_lt_gt_or_eq(input, expected):
    machine = IntcodeMachine(
        [
            3,
            21,
            1008,
            21,
            8,
            20,
            1005,
            20,
            22,
            107,
            8,
            21,
            20,
            1006,
            20,
            31,
            1106,
            0,
            36,
            98,
            0,
            0,
            1002,
            21,
            125,
            20,
            4,
            20,
            1105,
            1,
            46,
            104,
            999,
            1105,
            1,
            46,
            1101,
            1000,
            1,
            20,
            4,
            20,
            1105,
            1,
            46,
            98,
            99,
        ],
        iter((input,)),
    )
    [result] = list(machine.run())
    assert result == expected
