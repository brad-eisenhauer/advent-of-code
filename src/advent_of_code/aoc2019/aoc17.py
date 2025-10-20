"""Advent of Code 2019, day 17: https://adventofcode.com/2019/day/17"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(17, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        machine = IntcodeMachine(program)
        output = bytes(machine.run()).decode("ASCII")
        for line in output.split():
            log.debug(line)
        graph = Graph.from_strings(output.split())
        return calc_sum_of_alignment_parameters(graph)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        program[0] = 2
        control_lines = [
            "A,A,B,C,B,C,B,C,C,A",
            "R,8,L,4,R,4,R,10,R,8",
            "L,12,L,12,R,8,R,8",
            "R,10,R,4,R,4",
            "n",
        ]
        control_chars = list(control_feed(control_lines))
        machine = IntcodeMachine(program, input_stream=iter(control_chars))
        result = list(machine.run())
        return result[-1]


def control_feed(lines: Iterable[str]) -> Iterator[int]:
    for line in lines:
        yield from line.encode("ASCII")
        yield ord("\n")


def calc_sum_of_alignment_parameters(graph: Graph) -> int:
    result = 0
    for (x, y), neighbors in graph.nodes.items():
        if len(neighbors) == 4:
            result += x * y
    return result


Node = tuple[int, int]


class Graph:
    @classmethod
    def from_strings(cls, lines: Iterable[str]) -> Graph:
        result = cls()
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char in "#><^v":
                    result.add_node((col_idx, row_idx))
        return result

    def __init__(self):
        self.nodes: dict[Node, set[Node]] = {}

    def add_node(self, node: Node):
        if node in self.nodes:
            return
        self.nodes[node] = set()
        for offset in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbor: tuple[int, int] = tuple(a + b for a, b in zip(node, offset))
            if neighbor in self.nodes:
                self.nodes[node].add(neighbor)
                self.nodes[neighbor].add(node)


SAMPLE_OUTPUT = """\
..#..........
..#..........
#######...###
#.#...#...#.#
#############
..#...#...#..
..#####...^..
"""


def test_sum_of_alignment_parameters():
    g = Graph.from_strings(SAMPLE_OUTPUT.split())
    result = calc_sum_of_alignment_parameters(g)
    assert result == 76
