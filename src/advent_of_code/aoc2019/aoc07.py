"""Advent of Code 2019, Day 7: https://adventofcode.com/2019/day/7"""

from io import StringIO
from itertools import chain, permutations, tee

import pytest

from advent_of_code.base import Solution

from .intcode import IntcodeMachine


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        array = AmpArray(program)
        max_thrust = max(
            array.calc_thruster_signal(phase_setting)
            for phase_setting in permutations([0, 1, 2, 3, 4])
        )
        return max_thrust

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        loop = AmpLoop(program)
        max_thrust = max(
            loop.calc_thruster_signal(phase_setting)
            for phase_setting in permutations([5, 6, 7, 8, 9])
        )
        return max_thrust


class AmpArray:
    def __init__(self, program: list[int]):
        self.program = program

    def calc_thruster_signal(self, phase_setting: list[int]) -> int:
        amplifiers = []
        for phase_value in phase_setting:
            if amplifiers:
                input_stream = chain.from_iterable(([phase_value], amplifiers[-1].run()))
            else:
                input_stream = iter([phase_value, 0])
            amplifiers.append(IntcodeMachine(self.program.copy(), input_stream))
        [result] = amplifiers[-1].run()
        return result


class AmpLoop:
    def __init__(self, program: list[int]):
        self.program = program

    def calc_thruster_signal(self, phase_setting: list[int]) -> int:
        amplifiers = []
        for phase_value in phase_setting:
            if amplifiers:
                input_stream = chain.from_iterable(([phase_value], amplifiers[-1].run()))
            else:
                input_stream = None
            amplifiers.append(IntcodeMachine(self.program.copy(), input_stream))
        left, right = tee(amplifiers[-1].run())
        amplifiers[0].input_stream = chain.from_iterable(([phase_setting[0], 0], left))
        for result in right:
            ...
        return result


@pytest.mark.parametrize(
    ("program", "phase_setting", "expected"),
    [
        ("3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0", [4, 3, 2, 1, 0], 43210),
        (
            "3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0",
            [0, 1, 2, 3, 4],
            54321,
        ),
        (
            "3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,"
            "31,4,31,99,0,0,0",
            [1, 0, 4, 3, 2],
            65210,
        ),
    ],
)
def test_amp_array(program, phase_setting, expected):
    with StringIO(program) as f:
        program = IntcodeMachine.read_buffer(f)
    assert AmpArray(program).calc_thruster_signal(phase_setting) == expected


@pytest.mark.parametrize(
    ("program", "phase_setting", "expected"),
    [
        (
            "3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5",
            [9, 8, 7, 6, 5],
            139629729,
        ),
        (
            "3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,"
            "53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10",
            [9, 7, 8, 5, 6],
            18216,
        ),
    ],
)
def test_amp_loop(program, phase_setting, expected):
    with StringIO(program) as f:
        program = IntcodeMachine.read_buffer(f)
    assert AmpLoop(program).calc_thruster_signal(phase_setting) == expected
