"""Advent of Code 2023, day 15: https://adventofcode.com/2023/day/15"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import reduce
from io import StringIO
from typing import IO, Iterable, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            text = fp.read().strip()
        return sum(hash_init(fragment) for fragment in text.split(","))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            text = fp.read()
            instructions = [Instruction.from_str(s) for s in text.split(",")]
        boxes = initialize(instructions)
        return sum(b.calc_total_focusing_power() for b in boxes)


def hash_init(text: str) -> int:
    def _hash_iter(current, char) -> int:
        current += ord(char)
        current *= 17
        current %= 256
        return current

    return reduce(_hash_iter, text, 0)


@dataclass(frozen=True)
class Lens:
    label: str
    focal_length: int


@dataclass
class Box:
    id: int
    lenses: dict[str, tuple[int, Lens]] = field(default_factory=dict)
    next_index: int = 1

    def calc_total_focusing_power(self) -> int:
        result = 0
        for index, (_, lens) in enumerate(sorted(self.lenses.values())):
            result += (1 + self.id) * (1 + index) * lens.focal_length
        return result

    def insert(self, lens: Lens):
        if lens.label in self.lenses:
            index, _ = self.lenses[lens.label]
            self.lenses[lens.label] = index, lens
        else:
            self.lenses[lens.label] = self.next_index, lens
            self.next_index += 1

    def remove(self, label: str):
        if label in self.lenses:
            del self.lenses[label]


@dataclass
class Instruction:
    label: str
    op: str
    focal_length: Optional[int] = None

    @classmethod
    def from_str(cls, text: str) -> Instruction:
        pattern = r"([a-z]+)([-=])([0-9])?"
        match = re.match(pattern, text)
        if match is None:
            raise ValueError(f"'{text}' did not match the expected pattern.")
        label, op, fl = match.groups()
        fl = int(fl) if fl is not None else None
        return cls(label, op, fl)

    def execute(self, boxes: list[Box]):
        box_index = hash_init(self.label)
        match self.op:
            case "-":
                boxes[box_index].remove(self.label)
            case "=":
                boxes[box_index].insert(Lens(self.label, self.focal_length))


def initialize(instructions: Iterable[Instruction]) -> list[Box]:
    boxes = [Box(index) for index in range(256)]
    for instruction in instructions:
        instruction.execute(boxes)
    if log.isEnabledFor(logging.DEBUG):
        for box in boxes:
            if box.lenses:
                log.debug("%s", box)
    return boxes


SAMPLE_INPUTS = [
    """\
rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 1320


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 145
