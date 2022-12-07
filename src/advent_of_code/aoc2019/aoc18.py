"""Advent of Code 2019, day 18: https://adventofcode.com/2019/day/18"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from io import StringIO
from itertools import count
from typing import Any, Collection, Iterator, Optional, TextIO
from uuid import uuid4

import networkx as nx
import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar, GraphSimplifier

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(18, 2019, **kwargs)

    def solve_part_one(self) -> int:
        return self.solve_maze(subdivide=False)

    def solve_part_two(self) -> int:
        return self.solve_maze(subdivide=True)

    def solve_maze(self, subdivide: bool) -> int:
        with self.open_input() as f:
            maze = Maze.parse(f, subdivide)
        Simplifier(maze).simplify()
        initial_state = State(frozenset(maze.start))
        return Solver(maze).find_min_cost_to_goal(initial_state)


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
    drones: frozenset[Node]
    keys: frozenset[str] = frozenset()
    next_to_move: Optional[Node] = None

    def __lt__(self, other) -> bool:
        return len(self.keys) > len(other.keys)


@dataclass
class Maze:
    graph: nx.Graph = field(hash=False, compare=False)
    start: set[Node]
    keys: set[str]

    @classmethod
    def parse(cls, f: TextIO, subdivide: bool = False) -> Maze:
        nodes: dict[tuple[int, int], Node] = {}
        start: set[Node] = set()
        keys: set[str] = set()
        for y, line in enumerate(f.readlines()):
            for x, char in enumerate(line):
                match char:
                    case "#":
                        ...
                    case ".":
                        nodes[(x, y)] = Node()
                    case "@":
                        n = Node()
                        nodes[(x, y)] = n
                        start.add(n)
                    case key if key.islower():
                        nodes[(x, y)] = Node(key=key)
                        keys.add(key)
                    case door if door.isupper():
                        nodes[(x, y)] = Node(door=door)
        if not start:
            raise ValueError("No start node found.")
        graph = nx.Graph()
        graph.add_nodes_from(nodes.values())
        for x, y in nodes:
            for dx, dy in [(0, 1), (1, 0)]:
                x1 = x + dx
                y1 = y + dy
                if (x1, y1) in nodes:
                    graph.add_edge(nodes[(x, y)], nodes[(x1, y1)], weight=1)
        if subdivide:
            # start node and neighbors are removed; neighbors of removed nodes become new
            # starting nodes
            nodes_to_remove = [*start, *(n for s in start for n in graph.neighbors(s))]
            start = set(
                neighbor
                for node in nodes_to_remove
                for neighbor in graph.neighbors(node)
                if neighbor not in nodes_to_remove
            )
            graph.remove_nodes_from(nodes_to_remove)
        log.debug("Raw maze has %d nodes and %d keys.", len(graph.nodes), len(keys))
        return Maze(graph, start, keys)


class Simplifier(GraphSimplifier[Node]):
    def __init__(self, maze: Maze):
        super().__init__(maze.graph)
        self.maze = maze

    def is_protected(self, node: Node, mode: int) -> bool:
        # start and key nodes are always protected; door nodes are protected in hallways, not
        # dead ends.  All other nodes may be safely removed.
        if node in self.maze.start or node.key:
            return True
        if node.door:
            return mode == self.HALLWAY
        return False


class Solver(AStar[State]):
    def __init__(self, maze: Maze):
        self.maze = maze

    def is_goal_state(self, state: State) -> bool:
        return state.keys == self.maze.keys

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        """Given a state, generate valid successor states

        If the previously-moved drone did not acquire a new key or encounter a locked door,
        we will consider only successor states where the same drone moves again.
        """
        for drone in state.drones:
            if state.next_to_move and drone != state.next_to_move:
                continue
            for neighbor in self.maze.graph.neighbors(drone):
                next_to_move = neighbor
                if neighbor.can_open(state.keys):
                    cost = self.maze.graph.get_edge_data(drone, neighbor)["weight"]
                    keys = state.keys
                    if neighbor.key and neighbor.key not in keys:
                        next_to_move = None
                        keys = keys | {neighbor.key}
                    yield cost, State(
                        frozenset(neighbor if d is drone else d for d in state.drones),
                        keys,
                        next_to_move,
                    )
                else:
                    yield 0, dataclasses.replace(state, next_to_move=None)


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
    maze = Maze(graph, start={nodes[4]}, keys={"a", "b"})
    assert result == maze


def test_find_all_keys(sample_input):
    maze = Maze.parse(sample_input)
    Simplifier(maze).simplify()
    initial_state = State(frozenset(maze.start))
    assert Solver(maze).find_min_cost_to_goal(initial_state) == 8
