"""Advent of Code 2016, day 8: https://adventofcode.com/2016/day/8"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import IO, ClassVar


from advent_of_code.base import Solution
from advent_of_code.util import ocr


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2016, **kwargs)

    def solve_part_one(self) -> int:
        grid = DisplayGrid()
        with self.open_input() as fp:
            grid.mutate(fp)
        grid.print()
        return sum(1 for row in grid.grid for char in row if char == DisplayGrid.BLOCK)

    def solve_part_two(self) -> int:
        grid = DisplayGrid()
        with self.open_input() as fp:
            grid.mutate(fp)
        return ocr.to_string(["".join(row) for row in grid.grid], block_char=DisplayGrid.BLOCK)


@dataclass
class DisplayGrid:
    grid: list[list[str]] = field(default_factory=lambda: [[" "] * 50 for _ in range(6)])
    BLOCK: ClassVar[str] = "▒"

    def rect(self, a: int, b: int):
        for row, col in product(range(b), range(a)):
            self.grid[row][col] = self.BLOCK

    def rotate_row(self, y: int, by: int):
        row_len = len(self.grid[y])
        new_row = [self.grid[y][(x - by) % row_len] for x in range(row_len)]
        self.grid[y] = new_row

    def rotate_col(self, x: int, by: int):
        col_len = len(self.grid)
        new_col = [self.grid[(y - by) % col_len][x] for y in range(col_len)]
        for y in range(col_len):
            self.grid[y][x] = new_col[y]

    def print(self, out: IO = sys.stdout):
        for row in self.grid:
            out.write("".join(row) + "\n")

    def mutate(self, instructions: IO):
        rect_pattern = r"rect (?P<a>\d+)x(?P<b>\d+)"
        rot_row_pattern = r"rotate row y=(?P<y>\d+) by (?P<by>\d+)"
        rot_col_pattern = r"rotate column x=(?P<x>\d+) by (?P<by>\d+)"

        for line in instructions:
            if (match := re.match(rect_pattern, line)) is not None:
                a = int(match.groupdict()["a"])
                b = int(match.groupdict()["b"])
                self.rect(a, b)
            elif (match := re.match(rot_row_pattern, line)) is not None:
                y = int(match.groupdict()["y"])
                by = int(match.groupdict()["by"])
                self.rotate_row(y, by)
            elif (match := re.match(rot_col_pattern, line)) is not None:
                x = int(match.groupdict()["x"])
                by = int(match.groupdict()["by"])
                self.rotate_col(x, by)
            else:
                raise ValueError(f"'{line}' did not match any instruction.")


def test_grid():
    grid = DisplayGrid()
    grid.rect(3, 2)
    grid.rotate_col(1, 1)
    grid.rotate_row(0, 4)
    grid.rotate_col(1, 1)
    grid.print()
