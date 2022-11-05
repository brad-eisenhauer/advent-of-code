"""Advent of Code 2020, day 7: https://adventofcode.com/2020/day/7"""
import re
from collections import deque
from functools import cache
from io import StringIO
from typing import TextIO

import networkx as nx
import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            graph = build_graph(f)
        return count_distinct_can_contain(graph, "shiny gold")

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            graph = build_graph(f)
        return count_all_contents(graph, "shiny gold") - 1


def build_graph(f: TextIO) -> nx.DiGraph:
    result = nx.DiGraph()
    for line in f:
        container, contents = line.split(" contain ")
        _, container = parse_description(container)
        for content in contents.split(", "):
            qty, content = parse_description(content)
            result.add_edge(container, content, weight=qty)
    return result


def parse_description(bag: str) -> tuple[int, str]:
    pattern = re.compile(r"(\d* )?(.+) bags?")
    match = pattern.match(bag)
    qty, desc = match.groups()
    qty = 0 if qty is None else int(qty)
    return qty, desc.strip()


def count_distinct_can_contain(graph: nx.DiGraph, bag_description: str) -> int:
    frontier = deque([bag_description])
    containers = set()
    while frontier:
        node = frontier.pop()
        for container in graph.predecessors(node):
            if container not in containers:
                frontier.append(container)
                containers.add(container)
    return len(containers)


@cache
def count_all_contents(graph: nx.DiGraph, bag_description: str) -> int:
    result = 1  # result includes the current bag
    for contained in graph.successors(bag_description):
        edge_data = graph.get_edge_data(bag_description, contained)
        if (weight := edge_data["weight"]) > 0:
            result += weight * count_all_contents(graph, contained)
    return result


SAMPLE_INPUT = """\
light red bags contain 1 bright white bag, 2 muted yellow bags.
dark orange bags contain 3 bright white bags, 4 muted yellow bags.
bright white bags contain 1 shiny gold bag.
muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
dark olive bags contain 3 faded blue bags, 4 dotted black bags.
vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
faded blue bags contain no other bags.
dotted black bags contain no other bags.
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("light red bags", (0, "light red")),
        ("1 bright white bag", (1, "bright white")),
        ("2 muted yellow bags.", (2, "muted yellow")),
        ("no other bags.", (0, "no other")),
    ],
)
def test_parse_description(description, expected):
    assert parse_description(description) == expected


def test_count_distinct_can_contain(sample_input):
    graph = build_graph(sample_input)
    result = count_distinct_can_contain(graph, "shiny gold")
    assert result == 4


@pytest.mark.parametrize(
    ("bag", "expected"),
    [
        ("faded blue", 1),
        ("dotted black", 1),
        ("vibrant plum", 12),
        ("dark olive", 8),
        ("shiny gold", 33),
    ],
)
def test_count_all_contained(sample_input, bag, expected):
    graph = build_graph(sample_input)
    result = count_all_contents(graph, bag)
    assert result == expected
