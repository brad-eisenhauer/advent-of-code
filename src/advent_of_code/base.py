from enum import Enum
from typing import Generic, TypeVar

from advent_of_code.util import get_input_path

T = TypeVar("T")


class PuzzlePart(str, Enum):
    All = "all"
    One = "one"
    Two = "two"


class Solution(Generic[T]):
    def __init__(self, day: int, year: int):
        self.day = day
        self.year = year

    def open_input(self):
        path = get_input_path(self.day, self.year)
        return open(path)

    def solve_part(self, part: PuzzlePart) -> T:
        match part:
            case PuzzlePart.All:
                raise ValueError("Part argument must be 'one' or 'two'.")
            case PuzzlePart.One:
                return self.solve_part_one()
            case PuzzlePart.Two:
                return self.solve_part_two()
            case _:
                raise ValueError("Unrecognized part.")

    def solve_part_one(self) -> T:
        raise NotImplementedError("Puzzle part one not yet implemented.")

    def solve_part_two(self) -> T:
        raise NotImplementedError("Puzzle part two not yet implemented.")
