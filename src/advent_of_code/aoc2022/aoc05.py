"""Advent of Code 2022, day 5: https://adventofcode.com/2022/day/5"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[str, str]):
    def __init__(self, **kwargs):
        super().__init__(5, 2022, **kwargs)

    def solve_part_one(self) -> str:
        with self.open_input() as f:
            crate_stacks = CrateStacks.parse(f)
            for line in f:
                crate_stacks.execute(line)
        return crate_stacks.show_top_crates()

    def solve_part_two(self) -> str:
        with self.open_input() as f:
            crate_stacks = CrateStacks.parse(f)
            for line in f:
                crate_stacks.execute(line, model="9001")
        return crate_stacks.show_top_crates()


@dataclass
class CrateStacks:
    stacks: dict[str, list[str]]

    @classmethod
    def parse(cls, f: TextIO) -> CrateStacks:
        line = next(f)
        crate_label_idxs = range(1, len(line), 4)
        stacks = [[] for _ in crate_label_idxs]
        labels = []
        while line != "\n":
            if line.lstrip().startswith("["):
                for stack, idx in zip(stacks, crate_label_idxs):
                    if line[idx] != " ":
                        stack.insert(0, line[idx])
            else:
                labels = [line[idx] for idx in crate_label_idxs]
            line = next(f)
        labelled_stacks = dict(zip(labels, stacks))
        return CrateStacks(labelled_stacks)

    def execute(self, instruction: str, model: str = "9000"):
        instruction_pattern = re.compile(r"move (\d+) from (\d+) to (\d+)")
        qty, src, dest = instruction_pattern.match(instruction).groups()
        qty = int(qty)
        if model == "9000":
            for _ in range(qty):
                crate = self.stacks[src].pop(-1)
                self.stacks[dest].append(crate)
        elif model == "9001":
            section = self.stacks[src][-qty:]
            self.stacks[src] = self.stacks[src][:-qty]
            self.stacks[dest] += section

    def show_top_crates(self) -> str:
        return "".join(stack[-1] for stack in self.stacks.values())


SAMPLE_INPUTS = [
    """\
    [D]
[N] [C]
[Z] [M] [P]
 1   2   3

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_crate_stacks_parse(sample_input):
    expected = CrateStacks({"1": ["Z", "N"], "2": ["M", "C", "D"], "3": ["P"]})
    assert CrateStacks.parse(sample_input) == expected


@pytest.mark.parametrize(("model", "expected"), [("9000", "CMZ"), ("9001", "MCD")])
def test_execute(sample_input, model, expected):
    crate_stacks = CrateStacks.parse(sample_input)
    for line in sample_input:
        crate_stacks.execute(line, model=model)
    assert crate_stacks.show_top_crates() == expected
