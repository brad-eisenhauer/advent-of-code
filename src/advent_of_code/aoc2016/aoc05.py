"""Advent of Code 2016, day 5: https://adventofcode.com/2016/day/5"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from itertools import count, islice
from typing import Iterator

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(5, 2016, **kwargs)

    def solve_part_one(self) -> int:
        door_cracker = Cracker("reyedfim")
        result = "".join(islice((c for c, _ in door_cracker.generate_password_chars()), 8))
        return result

    def solve_part_two(self) -> int:
        result = [None] * 8
        door_cracker = Cracker("reyedfim")
        pass_chars = door_cracker.generate_password_chars()
        while None in result:
            pos, char = next(pass_chars)
            try:
                pos = int(pos)
                if pos in range(len(result)) and result[pos] is None:
                    result[pos] = char
            except ValueError:
                ...
        return "".join(result)


@dataclass
class Cracker:
    door_id: str

    def generate_password_chars(self) -> Iterator[str]:
        for n in count():
            door_hash = md5(f"{self.door_id}{n}".encode(), usedforsecurity=False)
            if door_hash.hexdigest().startswith("00000"):
                yield door_hash.hexdigest()[5:7]
