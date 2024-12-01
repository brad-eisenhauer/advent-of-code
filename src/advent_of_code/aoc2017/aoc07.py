"""Advent of Code 2017, day 7: https://adventofcode.com/2017/day/7"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[str, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> str:
        with input_file or self.open_input() as reader:
            tree = build_prog_tree(reader)
        return tree.name

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            root = build_prog_tree(reader)
        oob = find_out_of_balance(root)
        weight_counts = Counter(c.total_weight for c in oob.subprograms)
        correct_weight = next(w for w, c in weight_counts.items() if c > 1)
        wrong_weight_node = next(p for p in oob.subprograms if p.total_weight != correct_weight)
        weight_error = wrong_weight_node.total_weight - correct_weight
        return wrong_weight_node.weight - weight_error


PROGRAM_REGEX = (
    r"(?P<name>[a-z]+) "               # name followed by a space
    r"\((?P<weight>\d+)\)"             # weight surrounded by parens
    r"(?:"                             # start of non-capturing, optional group
        r" -> "                        # literal arrow
        r"(?P<subprogs>"               # start of subprograms
        r"[a-z]+"                      # first subprogram name
        r"(?:, [a-z]+)*"               # additional subprograms
        r")"                           # end of subprograms
    r")?"                              # end of optional group
)


@dataclass
class Program:
    name: str
    weight: int
    subprogram_names: list[str]
    subprograms: list[Program] = field(default_factory=list)
    parent: Program | None = None

    @classmethod
    def parse(cls, text: str) -> Program:
        result = re.match(PROGRAM_REGEX, text)
        if result is None:
            raise ValueError(f"Unable to parse '{text}'.")
        groupdict = result.groupdict()
        name = groupdict["name"]
        weight = int(groupdict["weight"])
        subprogs = []
        if groupdict["subprogs"]:
            subprogs.extend(s.strip() for s in groupdict["subprogs"].split(","))
        return cls(name=name, weight=weight, subprogram_names=subprogs)

    @cached_property
    def total_weight(self) -> int:
        substack_weight = sum(p.total_weight for p in self.subprograms)
        return self.weight + substack_weight

    @property
    def is_balanced(self) -> bool:
        substack_weights = [p.total_weight for p in self.subprograms]
        if not substack_weights:
            return True
        return all(x == substack_weights[0] for x in substack_weights[1:])


def build_prog_tree(reader: IO) -> Program:
    programs: dict[str, Program] = {}
    for line in reader:
        p = Program.parse(line)
        programs[p.name] = p
    log.debug(programs)
    for p in programs.values():
        for child_name in p.subprogram_names:
            child = programs[child_name]
            p.subprograms.append(child)
            child.parent = p
    p = next(iter(programs.values()))
    while p.parent is not None:
        p = p.parent
    return p


def find_out_of_balance(root: Program) -> Program | None:
    log.debug("%s is balanced: %s", root.name, root.is_balanced)
    if root.is_balanced:
        return None
    log.debug(
        "%s has children: %s",
        root.name,
        {c.name: c.total_weight for c in root.subprograms},
    )
    for child in root.subprograms:
        if not child.is_balanced:
            return find_out_of_balance(child)
    return root


SAMPLE_INPUTS = [
    """\
pbga (66)
xhth (57)
ebii (61)
havc (66)
ktlj (57)
fwft (72) -> ktlj, cntj, xhth
qoyq (66)
padx (45) -> pbga, havc, qoyq
tknk (41) -> ugml, padx, fwft
jptl (61)
ugml (68) -> gyxo, ebii, jptl
gyxo (61)
cntj (57)
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
    assert solution.solve_part_one(sample_input) == "tknk"


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 60
