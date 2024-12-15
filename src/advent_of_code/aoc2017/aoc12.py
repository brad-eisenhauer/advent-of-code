"""Advent of Code 2017, day 12: https://adventofcode.com/2017/day/12"""

from __future__ import annotations

from collections import defaultdict, deque
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            comms_graph = read_comms_graph(fp)
        zero_accessible = find_comm_group("0", comms_graph)
        return len(zero_accessible)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            comms_graph = read_comms_graph(fp)
        result = 0
        remaining_nodes = set(comms_graph.keys())
        while remaining_nodes:
            result += 1
            next_node = remaining_nodes.pop()
            next_group = find_comm_group(next_node, comms_graph)
            remaining_nodes -= next_group
        return result


def read_comms_graph(reader: IO) -> dict[str, list[str]]:
    result = defaultdict(list)
    for line in reader:
        src, dests = line.split(" <-> ")
        for dest in dests.strip().split(", "):
            result[src].append(dest)
    return result


def find_comm_group(start: str, comms_graph: dict[str, list[str]]) -> set[str]:
    result: set[str] = set()
    frontier = deque([start])
    while frontier:
        next_node = frontier.pop()
        result.add(next_node)
        for neighbor in comms_graph[next_node]:
            if neighbor not in result:
                frontier.appendleft(neighbor)
    return result


SAMPLE_INPUTS = [
    """\
0 <-> 2
1 <-> 1
2 <-> 0, 3, 4
3 <-> 2, 4
4 <-> 2, 3, 6
5 <-> 6
6 <-> 4, 5
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_read_comms_graph(sample_input: IO) -> None:
    assert read_comms_graph(sample_input) == {
        "0": ["2"],
        "1": ["1"],
        "2": ["0", "3", "4"],
        "3": ["2", "4"],
        "4": ["2", "3", "6"],
        "5": ["6"],
        "6": ["4", "5"],
    }


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 6


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 2
