"""Advent of Code 2019, day 18: https://adventofcode.com/2019/day/18"""
from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from io import StringIO
from itertools import count, product
from typing import Any, Collection, Optional, TextIO, Iterator
from uuid import uuid4

import networkx as nx
import pytest

from advent_of_code.base import Solution
from advent_of_code.util import PriorityQueue, Dijkstra

log = logging.getLogger("aoc")


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(18, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            maze = Maze.parse(f)
        maze.simplify()
        return maze.find_all_keys()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            maze = Maze2.parse(f)
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

    def is_goal_state(self, state: State) -> bool:
        return state.keys == self.keys

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        for neighbor in self.graph.neighbors(state.location):
            if not neighbor.can_open(state.keys):
                continue
            cost = self.graph.get_edge_data(state.location, neighbor)["weight"]
            keys = state.keys
            if neighbor.key:
                keys = keys | {neighbor.key}
            yield cost, State(neighbor, keys)

    def find_all_keys(self) -> int:
        solver = Dijkstra(self.generate_next_states, self.is_goal_state)
        initial_state = State(self.start, frozenset())
        return solver.find_min_cost_to_goal(initial_state)


@dataclass(frozen=True)
class State2:
    locations: tuple[Node, ...]
    keys: frozenset[str]
    next_to_move: Optional[Node] = None

    def __lt__(self, other) -> bool:
        return len(self.keys) > len(other.keys)


@dataclass
class Maze2:
    graph: nx.Graph
    state: State2
    keys: set[str]

    @classmethod
    def parse(cls, f: TextIO) -> Maze2:
        lines = f.readlines()

        def modify_maze():
            for y, line in enumerate(lines):
                for x, char in enumerate(line):
                    if char == "@":
                        lines[y] = re.sub(r".@.", "###", lines[y])
                        lines[y - 1] = [
                            "@" if abs(x - x1) == 1 else "#" if x == x1 else c
                            for x1, c in enumerate(lines[y - 1])
                        ]
                        lines[y + 1] = [
                            "@" if abs(x - x1) == 1 else "#" if x == x1 else c
                            for x1, c in enumerate(lines[y + 1])
                        ]
                        return

        modify_maze()
        start_locs: list[Node] = []
        nodes: dict[tuple[int, int], Node] = {}
        keys = set()
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                match char:
                    case "#":
                        ...
                    case ".":
                        nodes[(x, y)] = Node()
                    case "@":
                        nodes[(x, y)] = Node()
                        start_locs.append(nodes[(x, y)])
                    case key if key.islower():
                        nodes[(x, y)] = Node(key=key)
                        keys |= {key}
                    case door if door.isupper():
                        nodes[(x, y)] = Node(door=door)
        graph = nx.Graph()
        graph.add_nodes_from(nodes.values())
        for x, y in nodes:
            for dx, dy in [(0, 1), (1, 0)]:
                x1 = x + dx
                y1 = y + dy
                if (x1, y1) in nodes:
                    graph.add_edge(nodes[(x, y)], nodes[(x1, y1)], weight=1)
        log.debug("Raw maze has %d nodes.", len(nodes))
        return Maze2(graph, State2(tuple(start_locs), frozenset()), keys)

    def simplify(self, depth: int = 1):
        g = self.graph
        nodes_to_remove = []
        for node in g.nodes:
            if node in self.state.locations or node.key:
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

    def is_goal_state(self, state: State2) -> bool:
        return state.keys == self.keys

    def generate_next_states(self, state: State2) -> Iterator[tuple[int, State2]]:
        for drone in state.locations:
            if state.next_to_move and drone != state.next_to_move:
                continue
            for neighbor in self.graph.neighbors(drone):
                next_to_move = neighbor
                if neighbor.can_open(state.keys):
                    cost = self.graph.get_edge_data(drone, neighbor)["weight"]
                    keys = state.keys
                    if neighbor.key:
                        next_to_move = None
                        keys = keys | {neighbor.key}
                    yield cost, State2(tuple(neighbor if d is drone else d for d in state.locations), keys, next_to_move)
                else:
                    yield 0, dataclasses.replace(state, next_to_move=None)

    def find_all_keys(self) -> int:
        solver = Dijkstra(self.generate_next_states, self.is_goal_state)
        return solver.find_min_cost_to_goal(self.state)


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
