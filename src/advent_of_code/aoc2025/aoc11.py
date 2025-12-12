"""Advent of Code 2025, day 11: https://adventofcode.com/2025/day/11"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import cache
from io import StringIO
from typing import IO, ClassVar, Optional, Self

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            network = Network.read(fp)
        return network.count_paths("you")

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            network = Network.read(fp)
        if fft_to_dac := network.count_paths("fft", "dac"):
            return (
                network.count_paths("svr", "fft") * fft_to_dac * network.count_paths("dac", "out")
            )
        return (
            network.count_paths("svr", "dac")
            * network.count_paths("dac", "fft")
            * network.count_paths("fft", "out")
        )


@dataclass(frozen=True)
class Network:
    nodes: dict[str, list[str]] = field(hash=False)

    PATTERN: ClassVar[str] = r"(?P<node>[a-z]+): (?P<outputs>[a-z ]+)"

    @classmethod
    def read(cls, reader: IO) -> Self:
        nodes = {}
        for line in reader:
            if (match := re.match(cls.PATTERN, line)) is None:
                raise ValueError(f"No match for '{line.strip()}'")
            match_groups = match.groupdict()
            nodes[match_groups["node"]] = match_groups["outputs"].split()
        return cls(nodes)

    @cache
    def count_paths(self, start: str, end: str = "out") -> int:
        if start == end:
            return 1
        if start not in self.nodes:
            return 0
        result = 0
        for output in self.nodes[start]:
            result += self.count_paths(output, end)
        return result


SAMPLE_INPUTS = [
    """\
aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out
""",
    """\
svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 5


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 2
