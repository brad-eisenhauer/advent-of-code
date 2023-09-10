"""Advent of Code 2015, day 16: https://adventofcode.com/2015/day/16"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from advent_of_code.base import Solution


MFCSAM = {
    "children": 3,
    "cats": 7,
    "samoyeds": 2,
    "pomeranians": 3,
    "akitas": 0,
    "vizslas": 0,
    "goldfish": 5,
    "trees": 3,
    "cars": 2,
    "perfumes": 1,
}


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            for line in f:
                a = Aunt.from_str(line.strip())
                if a.matches(MFCSAM):
                    return a.id

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            for line in f:
                a = Aunt.from_str(line.strip())
                if a.matches(MFCSAM, modified=True):
                    return a.id


@dataclass
class Aunt:
    id: int
    children: Optional[int] = None
    cats: Optional[int] = None
    samoyeds: Optional[int] = None
    pomeranians: Optional[int] = None
    akitas: Optional[int] = None
    vizslas: Optional[int] = None
    goldfish: Optional[int] = None
    trees: Optional[int] = None
    cars: Optional[int] = None
    perfumes: Optional[int] = None

    @classmethod
    def from_str(cls, text: str) -> Aunt:
        pattern = r"^Sue (?P<id>\d+): (?P<properties>.+)$"
        match_ = re.match(pattern, text).groupdict()
        id = int(match_["id"])
        props = {}
        for prop in match_["properties"].split(", "):
            key, val = prop.split(": ")
            props[key] = int(val)
        return Aunt(id, **props)

    def matches(self, properties: dict, modified: bool = False) -> bool:
        for key, value in properties.items():
            my_value = getattr(self, key)
            if my_value is None:
                continue
            if not modified:
                if my_value != value:
                    return False
            elif key in ("cats", "trees"):
                if my_value <= value:
                    return False
            elif key in ("pomeranians", "goldfish"):
                if my_value >= value:
                    return False
            elif my_value != value:
                return False
        return True
