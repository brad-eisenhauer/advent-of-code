"""Advent of Code 2016, day 2: https://adventofcode.com/2016/day/2"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import IO, Iterator, Optional

from advent_of_code.base import Solution

Vector = tuple[int, ...]


class AocSolution(Solution[str, str]):
    def __init__(self, **kwargs):
        super().__init__(2, 2016, **kwargs)

    def solve_part_one(self) -> str:
        keypad = Keypad.from_grid(KEYPAD_1)
        with self.open_input() as fp:
            return keypad.get_code((1, 1), fp)

    def solve_part_two(self) -> str:
        keypad = Keypad.from_grid(KEYPAD_2)
        with self.open_input() as fp:
            return keypad.get_code((2, 0), fp)


KEYPAD_1 = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["7", "8", "9"],
]
KEYPAD_2 = [
    [None, None, "1", None, None],
    [None, "2", "3", "4", None],
    ["5", "6", "7", "8", "9"],
    [None, "A", "B", "C", None],
    [None, None, "D", None, None],
]
MOVEMENTS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass
class Keypad:
    buttons: list[Optional[str]]
    dim: int

    @classmethod
    def from_grid(cls, keypad: list[list[Optional[str]]]) -> Keypad:
        buttons = list(chain(*keypad))
        dim = len(keypad)
        return cls(buttons, dim)

    def read(self, pos: Vector) -> Optional[str]:
        row, col = pos
        return self.buttons[self.dim * row + col]

    def move(self, start: Vector, instruction: str) -> Vector:
        def _chunk_instruction() -> Iterator[tuple[str, int]]:
            last_char: Optional[str] = None
            char_count = 0
            for char in instruction:
                if char != last_char:
                    if last_char:
                        yield last_char, char_count
                    last_char = char
                    char_count = 1
                else:
                    char_count += 1
            yield last_char, char_count

        pos = start
        for movement_char, count in _chunk_instruction():
            new_pos = tuple(
                clamp(a + count * b, self.dim) for a, b in zip(pos, MOVEMENTS[movement_char])
            )
            if self.read(new_pos):
                pos = new_pos
        return pos

    def get_code(self, start: Vector, instructions: IO) -> str:
        result = ""
        pos = start
        for line in instructions:
            pos = self.move(pos, line.strip())
            result += self.read(pos)
        return result


def clamp(value: int, r: int) -> int:
    result = max(0, value)
    result = min(r - 1, result)
    return result
