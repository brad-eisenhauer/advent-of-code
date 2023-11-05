"""Advent of Code 2019, day 20: https://adventofcode.com/2019/day/20"""
from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from io import StringIO
from typing import Iterator, Optional, TextIO

import networkx as nx
import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar, GraphSimplifier

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(20, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            maze = Maze.parse(f)
        Simplifier(maze).simplify()
        return Solver(maze).find_min_cost_to_goal(State(maze.start))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            maze = Maze.parse(f)
        Simplifier(maze).simplify()
        return Solver(maze, recursive=True).find_min_cost_to_goal(State(maze.start))


class Edge(Enum):
    Inner = auto()
    Outer = auto()


@dataclass(frozen=True)
class Node:
    id: tuple[int, int]
    tag: Optional[str] = None
    edge: Optional[Edge] = None


@dataclass(frozen=True)
class State:
    location: Node
    depth: int = 0

    def __lt__(self, other):
        return self.location.id < other.location.id


class Maze:
    def __init__(self, graph: nx.Graph, start: Node):
        self.graph = graph
        self.start = start

        self.tags: dict[str, list[Node]] = defaultdict(list)
        for node in graph.nodes:
            if node.tag:
                self.tags[node.tag].append(node)

    @classmethod
    def parse(cls, f: TextIO) -> Maze:
        lines = f.readlines()
        center = len(lines[0]) // 2, len(lines) // 2
        # find tags
        tags: dict[tuple[int, int], tuple[str, Edge]] = {}
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char.isupper():
                    # assume char is the start of a tag; search 4 directions for tag and applicable node
                    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    for dx, dy in directions:
                        try:
                            char2 = lines[y + dy][x + dx]
                            node_char = lines[y + 2 * dy][x + 2 * dx]
                        except IndexError:
                            ...
                        else:
                            if char2.isupper() and node_char == ".":
                                tag = char + char2 if min(dx, dy) == 0 else char2 + char
                                node_x = x + 2 * dx
                                node_y = y + 2 * dy
                                if dx:  # horizontal
                                    edge = (
                                        Edge.Inner
                                        if (node_x > center[0]) + min(dx, dy)
                                        else Edge.Outer
                                    )
                                else:  # vertical
                                    edge = (
                                        Edge.Inner
                                        if (node_y > center[1]) + min(dx, dy)
                                        else Edge.Outer
                                    )
                                tags[(node_x, node_y)] = tag, edge

        # create nodes
        start: Optional[Node] = None
        nodes: dict[tuple[int, int], Node] = {}
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                match char:
                    case ".":
                        tag, edge = tags.get((x, y), (None, None))
                        nodes[(x, y)] = Node((x, y), tag, edge)
                        if tag == "AA":
                            start = nodes[(x, y)]
                    case _:
                        ...

        # create graph
        graph = nx.Graph()
        graph.add_nodes_from(nodes.values())
        for (x, y), node in nodes.items():
            for dx, dy in [(0, 1), (1, 0)]:
                x1 = x + dx
                y1 = y + dy
                if (x1, y1) in nodes:
                    graph.add_edge(node, nodes[(x1, y1)], weight=1)

        log.debug("Raw maze has %d nodes.", len(graph.nodes))
        return Maze(graph, start)


class Simplifier(GraphSimplifier[Node]):
    def __init__(self, maze: Maze):
        super().__init__(maze.graph)

    def is_protected(self, node: Node, mode: int) -> bool:
        return node.tag is not None


class Solver(AStar[State]):
    def __init__(self, maze: Maze, recursive: bool = False):
        self.maze = maze
        self.recursive = recursive

    def is_goal_state(self, state: State) -> bool:
        return state.depth == 0 and state.location.tag == "ZZ"

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        g = self.maze.graph
        loc = state.location
        for neighbor in g.neighbors(loc):
            cost = g.get_edge_data(loc, neighbor)["weight"]
            yield cost, dataclasses.replace(state, location=neighbor)
        if loc.tag:
            for node in self.maze.tags[loc.tag]:
                if node is not loc:
                    depth = state.depth + (1 if loc.edge is Edge.Inner else -1)
                    new_state = State(node, depth if self.recursive else 0)
                    if new_state.depth >= 0:
                        yield 1, new_state


SAMPLE_INPUTS = [
    """\
         A
         A
  #######.#########
  #######.........#
  #######.#######.#
  #######.#######.#
  #######.#######.#
  #####  B    ###.#
BC...##  C    ###.#
  ##.##       ###.#
  ##...DE  F  ###.#
  #####    G  ###.#
  #########.#####.#
DE..#######...###.#
  #.#########.###.#
FG..#########.....#
  ###########.#####
             Z
             Z
""",
    """\
                   A
                   A
  #################.#############
  #.#...#...................#.#.#
  #.#.#.###.###.###.#########.#.#
  #.#.#.......#...#.....#.#.#...#
  #.#########.###.#####.#.#.###.#
  #.............#.#.....#.......#
  ###.###########.###.#####.#.#.#
  #.....#        A   C    #.#.#.#
  #######        S   P    #####.#
  #.#...#                 #......VT
  #.#.#.#                 #.#####
  #...#.#               YN....#.#
  #.###.#                 #####.#
DI....#.#                 #.....#
  #####.#                 #.###.#
ZZ......#               QG....#..AS
  ###.###                 #######
JO..#.#.#                 #.....#
  #.#.#.#                 ###.#.#
  #...#..DI             BU....#..LF
  #####.#                 #.#####
YN......#               VT..#....QG
  #.###.#                 #.###.#
  #.#...#                 #.....#
  ###.###    J L     J    #.#.###
  #.....#    O F     P    #.#...#
  #.###.#####.#.#####.#####.###.#
  #...#.#.#...#.....#.....#.#...#
  #.#####.###.###.#.#.#########.#
  #...#.#.....#...#.#.#.#.....#.#
  #.###.#####.###.###.#.#.#######
  #.#.........#...#.............#
  #########.###.###.#############
           B   J   C
           U   P   P
""",
    """\
             Z L X W       C
             Z P Q B       K
  ###########.#.#.#.#######.###############
  #...#.......#.#.......#.#.......#.#.#...#
  ###.#.#.#.#.#.#.#.###.#.#.#######.#.#.###
  #.#...#.#.#...#.#.#...#...#...#.#.......#
  #.###.#######.###.###.#.###.###.#.#######
  #...#.......#.#...#...#.............#...#
  #.#########.#######.#.#######.#######.###
  #...#.#    F       R I       Z    #.#.#.#
  #.###.#    D       E C       H    #.#.#.#
  #.#...#                           #...#.#
  #.###.#                           #.###.#
  #.#....OA                       WB..#.#..ZH
  #.###.#                           #.#.#.#
CJ......#                           #.....#
  #######                           #######
  #.#....CK                         #......IC
  #.###.#                           #.###.#
  #.....#                           #...#.#
  ###.###                           #.#.#.#
XF....#.#                         RF..#.#.#
  #####.#                           #######
  #......CJ                       NM..#...#
  ###.#.#                           #.###.#
RE....#.#                           #......RF
  ###.###        X   X       L      #.#.#.#
  #.....#        F   Q       P      #.#.#.#
  ###.###########.###.#######.#########.###
  #.....#...#.....#.......#...#.....#.#...#
  #####.#.###.#######.#######.###.###.#.#.#
  #.......#.......#.#.#.#.#...#...#...#.#.#
  #####.###.#####.#.#.#.#.###.###.#.###.###
  #.......#.....#.#...#...............#...#
  #############.#.#.###.###################
               A O F   N
               A A D   M
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "recursive", "expected"),
    [(0, False, 23), (1, False, 58), (0, True, 26), (2, True, 396)],
    indirect=["sample_input"],
)
def test_calc_distance(sample_input, recursive, expected):
    maze = Maze.parse(sample_input)
    Simplifier(maze).simplify()
    solver = Solver(maze, recursive=recursive)
    initial_state = State(maze.start)
    assert solver.find_min_cost_to_goal(initial_state) == expected
