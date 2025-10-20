"""Advent of Code 2022, day 10: https://adventofcode.com/2022/day/10"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from itertools import islice
from typing import Callable, Iterable, Iterator

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.ocr import to_string


class AocSolution(Solution[int, str]):
    def __init__(self, **kwargs):
        super().__init__(10, 2022, **kwargs)

    def solve_part_one(self) -> int:
        cpu = CPU()

        def filter(t):
            return t[0] % 40 == 19

        with self.open_input() as f:
            return sum_signal_strengths(cpu.run_filter(f, filter), n=6)

    def solve_part_two(self) -> str:
        cpu = CPU()
        lines = []
        with self.open_input() as f:
            output = cpu.run(f)
            while True:
                line = build_line(islice(output, 40))
                if not line.strip():
                    break
                lines.append(line)
        return to_string(lines)


@dataclass
class CPU:
    x: int = 1
    cycle_count: int = 0

    def execute(self, command: str):
        match command.split():
            case ["noop"]:
                self.cycle_count += 1
            case ["addx", n]:
                self.x += int(n)
                self.cycle_count += 2

    @staticmethod
    def _calc_command_cycle_count(command: str) -> int:
        match command.split()[0]:
            case "noop":
                return 1
            case "addx":
                return 2

    def run(self, commands: Iterable[str]) -> Iterator[tuple[int, int]]:
        yield self.cycle_count, self.x
        for line in commands:
            for i in range(self._calc_command_cycle_count(line) - 1):
                yield self.cycle_count + i + 1, self.x
            self.execute(line)
            yield self.cycle_count, self.x

    def run_filter(
        self, commands: Iterable[str], predicate: Callable[[tuple[int, int]], bool]
    ) -> Iterator[tuple[int, int]]:
        yield from filter(predicate, self.run(commands))


def sum_signal_strengths(cpu_states: Iterator[tuple[int, int]], n: int = 6) -> int:
    return sum(x * (cycle_count + 1) for cycle_count, x in islice(cpu_states, n))


def build_line(states: Iterable[tuple[int, int]]) -> str:
    def _generate() -> Iterator[str]:
        for cycle_count, x in states:
            yield "▒" if abs(x - cycle_count % 40) < 2 else " "

    return "".join(_generate())


SAMPLE_INPUTS = [
    """\
noop
addx 3
addx -5
""",
    """\
addx 15
addx -11
addx 6
addx -3
addx 5
addx -1
addx -8
addx 13
addx 4
noop
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx -35
addx 1
addx 24
addx -19
addx 1
addx 16
addx -11
noop
noop
addx 21
addx -15
noop
noop
addx -3
addx 9
addx 1
addx -3
addx 8
addx 1
addx 5
noop
noop
noop
noop
noop
addx -36
noop
addx 1
addx 7
noop
noop
noop
addx 2
addx 6
noop
noop
noop
noop
noop
addx 1
noop
noop
addx 7
addx 1
noop
addx -13
addx 13
addx 7
noop
addx 1
addx -33
noop
noop
noop
addx 2
noop
noop
noop
addx 8
noop
addx -1
addx 2
addx 1
noop
addx 17
addx -9
addx 1
addx 1
addx -3
addx 11
noop
noop
addx 1
noop
addx 1
noop
noop
addx -13
addx -19
addx 1
addx 3
addx 26
addx -30
addx 12
addx -1
addx 3
addx 1
noop
noop
noop
addx -9
addx 18
addx 1
addx 2
noop
noop
addx 9
noop
noop
noop
addx -1
addx 2
addx -37
addx 1
addx 3
noop
addx 15
addx -21
addx 22
addx -6
addx 1
noop
addx 2
addx 1
noop
addx -10
noop
noop
addx 20
addx 1
addx 2
addx 2
addx -6
addx -11
noop
noop
noop
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(("sample_input", "expected"), [(0, (5, -1))], indirect=["sample_input"])
def test_run(sample_input, expected):
    cpu = CPU()
    [*_, final_state] = cpu.run(sample_input)
    assert final_state == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(1, [(19, 21), (59, 19), (99, 18), (139, 21), (179, 16), (219, 18)])],
    indirect=["sample_input"],
)
def test_run_filter(sample_input, expected):
    cpu = CPU()
    filtered_states = list(cpu.run_filter(sample_input, lambda t: t[0] % 40 == 19))
    assert filtered_states == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(1, 13140)],
    indirect=["sample_input"],
)
def test_sum_signal_strengths(sample_input, expected):
    cpu = CPU()
    result = sum_signal_strengths(cpu.run_filter(sample_input, lambda t: t[0] % 40 == 19), n=6)
    assert result == expected
