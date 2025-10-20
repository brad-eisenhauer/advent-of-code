"""Advent of Code 2015, day 8: https://adventofcode.com/2015/day/8"""

from __future__ import annotations

from ast import literal_eval

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2015, **kwargs)

    def solve_part_one(self) -> int:
        result = 0
        with self.open_input() as f:
            for line in f:
                line = line.rstrip()
                result += len(line) - len(literal_eval(line))
        return result

    def solve_part_two(self) -> int:
        result = 0
        with self.open_input() as f:
            for line in f:
                line = line.rstrip()
                result += len(line.replace("\\", "\\\\").replace('"', '\\"')) + 2 - len(line)
        return result
