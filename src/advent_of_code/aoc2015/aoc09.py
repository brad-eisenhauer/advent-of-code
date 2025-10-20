"""Advent of Code 2015, day 9: https://adventofcode.com/2015/day/9"""

from __future__ import annotations

from io import StringIO
from itertools import combinations
from typing import Iterator, Optional, TextIO

import networkx as nx
import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            routes = parse_routes(f)
        return find_shortest_route(routes)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            routes = parse_routes(f)
        return find_longest_route(routes)


def parse_routes(f: TextIO) -> nx.Graph:
    result = nx.Graph()
    for line in f:
        route, dist = line.split(" = ")
        left, right = route.split(" to ")
        result.add_edge(left, right, weight=int(dist))
    return result


def find_shortest_route(routes: nx.Graph) -> int:
    shortest_route: Optional[int] = None
    for start, end in combinations(routes.nodes, 2):
        nav = Navigator(routes, end)
        dist = nav.find_min_cost_to_goal((start, frozenset({start})))
        if shortest_route is None or dist < shortest_route:
            shortest_route = dist
    return shortest_route


def find_longest_route(routes: nx.Graph) -> int:
    min_miles_dropped: Optional[int] = None
    for start, end in combinations(routes.nodes, 2):
        nav = Navigator(routes, end, invert=True)
        dist_dropped = nav.find_min_cost_to_goal((start, frozenset({start})))
        if min_miles_dropped is None or dist_dropped < min_miles_dropped:
            min_miles_dropped = dist_dropped
    max_dist = (len(routes.nodes) - 1) * max(
        routes.get_edge_data(left, right)["weight"] for left, right in routes.edges
    )
    return max_dist - min_miles_dropped


TravelState = tuple[str, frozenset[str]]


class Navigator(AStar[TravelState]):
    def __init__(self, graph: nx.Graph, end: str, invert: bool = False):
        self._graph = graph
        self._end = end
        self._invert = invert
        self._max_edge_weight = max(
            graph.get_edge_data(left, right)["weight"] for left, right in graph.edges
        )

    def is_goal_state(self, state: TravelState) -> bool:
        _, visited = state
        return len(visited) == len(self._graph.nodes)

    def generate_next_states(self, state: TravelState) -> Iterator[tuple[int, TravelState]]:
        current, visited = state
        for neighbor in self._graph.neighbors(current):
            if neighbor in visited:
                continue
            cost = self._graph.get_edge_data(current, neighbor)["weight"]
            if self._invert:
                cost = self._max_edge_weight - cost
            yield cost, (neighbor, visited | {neighbor})


SAMPLE_INPUTS = [
    """\
London to Dublin = 464
London to Belfast = 518
Dublin to Belfast = 141
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.skip("Can't compare graphs.")
def test_parse_routes(sample_input):
    expected = nx.Graph()
    expected.add_nodes_from(["London", "Dublin", "Belfast"])
    expected.add_edge("London", "Dublin", weight=464)
    expected.add_edge("London", "Belfast", weight=518)
    expected.add_edge("Dublin", "Belfast", weight=141)

    assert parse_routes(sample_input) == expected


def test_find_shortest_route(sample_input):
    routes = parse_routes(sample_input)
    result = find_shortest_route(routes)
    assert result == 605


def test_find_longest_route(sample_input):
    routes = parse_routes(sample_input)
    result = find_longest_route(routes)
    assert result == 982
