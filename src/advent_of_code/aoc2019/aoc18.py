"""Advent of Code 2019, day 18: https://adventofcode.com/2019/day/18"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import StringIO
from itertools import count
from typing import Any, Collection, Optional, TextIO
from uuid import uuid4

import networkx as nx
import pytest

from advent_of_code.base import Solution
from advent_of_code.util import PriorityQueue

log = logging.getLogger("aoc")


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(18, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            maze = Maze.parse(f)
        maze.simplify()
        return maze.find_all_keys()


@dataclass(frozen=True)
class Node:
    id: Any = field(default_factory=lambda: uuid4())
    key: Optional[str] = None
    door: Optional[str] = None

    def can_open(self, keys: Collection[str]) -> bool:
        if self.door is None:
            return True
        return self.door.lower() in keys


@dataclass(frozen=True)
class State:
    location: Node
    keys: frozenset[str]

    def __lt__(self, other) -> bool:
        return len(self.keys) > len(other.keys)


@dataclass(frozen=True)
class Maze:
    graph: nx.Graph = field(hash=False, compare=False)
    start: Node
    keys: set[str]

    @classmethod
    def parse(cls, f: TextIO) -> Maze:
        nodes: dict[tuple[int, int], Node] = {}
        start: Optional[Node] = None
        keys: set[str] = set()
        for y, line in enumerate(f.readlines()):
            for x, char in enumerate(line):
                match char:
                    case "#":
                        ...
                    case ".":
                        nodes[(x, y)] = Node()
                    case "@":
                        nodes[(x, y)] = Node()
                        start = nodes[(x, y)]
                    case key if key.islower():
                        nodes[(x, y)] = Node(key=key)
                        keys.add(key)
                    case door if door.isupper():
                        nodes[(x, y)] = Node(door=door)
        if start is None:
            raise ValueError("No start node found.")
        graph = nx.Graph()
        graph.add_nodes_from(nodes.values())
        for x, y in nodes:
            for dx, dy in [(0, 1), (1, 0)]:
                x1 = x + dx
                y1 = y + dy
                if (x1, y1) in nodes:
                    graph.add_edge(nodes[(x, y)], nodes[(x1, y1)], weight=1)
        log.debug("Raw maze has %d nodes.", len(nodes))
        return Maze(graph, start, keys)

    def simplify(self, depth: int = 1):
        g = self.graph
        nodes_to_remove = []
        for node in g.nodes:
            if node is self.start or node.key:
                continue
            neighbors = list(g.neighbors(node))
            match len(neighbors), node.door:
                case 1, _:
                    g.remove_edge(node, neighbors[0])
                    nodes_to_remove.append(node)
                case 2, None:
                    weight = sum(g.get_edge_data(node, n)["weight"] for n in neighbors)
                    g.add_edge(*neighbors, weight=weight)
                    for n in neighbors:
                        g.remove_edge(n, node)
                    nodes_to_remove.append(node)
                case _:
                    ...
        if nodes_to_remove:
            g.remove_nodes_from(nodes_to_remove)
            self.simplify(depth + 1)
        else:
            log.debug("Ran simplify %d times.", depth)
            log.debug("Simplified maze has %d nodes.", len(g.nodes))

    def find_all_keys(self) -> int:
        initial_state = State(self.start, frozenset())
        visited_states = {initial_state: 0}
        frontier = PriorityQueue()
        frontier.push(0, initial_state)
        while frontier:
            current_state = frontier.pop()
            for neighbor in self.graph.neighbors(current_state.location):
                if not neighbor.can_open(current_state.keys):
                    continue
                step_count = (
                    visited_states[current_state]
                    + self.graph.get_edge_data(current_state.location, neighbor)["weight"]
                )
                keys = current_state.keys
                if neighbor.key:
                    keys = keys | {neighbor.key}
                if keys == self.keys:
                    return step_count
                next_state = State(neighbor, keys)
                if next_state not in visited_states or step_count < visited_states[next_state]:
                    visited_states[next_state] = step_count
                    frontier.push(step_count, next_state)


SAMPLE_INPUT = """\
#########
#b.A.@.a#
#########
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_parse_maze(sample_input, monkeypatch):
    ids = count(1)
    monkeypatch.setattr(__name__ + ".uuid4", lambda: next(ids))
    result = Maze.parse(sample_input)
    graph = nx.Graph()
    nodes = [
        Node(1, key="b"),
        Node(2),
        Node(3, door="A"),
        Node(4),
        Node(5),
        Node(6),
        Node(7, key="a"),
    ]
    graph.add_nodes_from(nodes)
    for left, right in zip(nodes, nodes[1:]):
        graph.add_edge(left, right)
    maze = Maze(graph, start=nodes[4], keys={"a", "b"})
    assert result == maze


def test_find_all_keys(sample_input):
    maze = Maze.parse(sample_input)
    maze.simplify()
    assert maze.find_all_keys() == 8
