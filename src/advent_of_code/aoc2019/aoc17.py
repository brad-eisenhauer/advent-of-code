"""Advent of Code 2019, day 17: https://adventofcode.com/2019/day/17"""
from __future__ import annotations

from itertools import product
from typing import Iterable

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(17, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        machine = IntcodeMachine(program)
        output = bytes(machine.run()).decode("ASCII")
        graph = Graph.from_strings(output.split())
        return calc_sum_of_alignment_parameters(graph)


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
