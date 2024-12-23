"""Advent of Code 2024, day 23: https://adventofcode.com/2024/day/23"""

from __future__ import annotations

import random
from io import StringIO
from itertools import combinations
from typing import IO, Optional

import networkx as nx
import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, str]):
    def __init__(self, **kwargs):
        super().__init__(23, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            network = read_network(reader)
        return count_connected_triads(network)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            network = read_network(reader)
        largest_group = find_largest_connected_group(network)
        return ",".join(sorted(largest_group))


def read_network(reader: IO) -> nx.Graph[str]:
    result = nx.Graph()
    for line in reader:
        left_node, right_node = line.strip().split("-")
        result.add_nodes_from([left_node, right_node])
        result.add_edge(left_node, right_node)
    return result


def count_connected_triads(network: nx.Graph[str]) -> int:
    results: set[frozenset[str]] = set()
    t_nodes = [node for node in network.nodes if node.startswith("t")]
    # for each t-node, loop through pairs of neighbors to see if they're connected.
    for node in t_nodes:
        for left, right in combinations(network.neighbors(node), 2):
            if network.has_edge(left, right):
                results.add(frozenset([node, left, right]))
    return len(results)


def find_largest_connected_group(network: nx.Graph[str]) -> set[str]:
    # build a maximal clique
    largest_clique: set[str] | None = None
    while largest_clique is None or len(largest_clique) < network.number_of_nodes():
        seed = random.choice(list(network.nodes))
        neighbors = network.neighbors(seed)
        try:
            clique = set([seed, next(neighbors)])
        except StopIteration:
            network.remove_node(seed)
            continue
        for neighbor in neighbors:
            if all(network.has_edge(neighbor, node) for node in clique):
                clique.add(neighbor)
        if largest_clique is None or len(clique) > len(largest_clique):
            largest_clique = clique
        network.remove_nodes_from(clique)
    if largest_clique is None:
        raise ValueError(("Well, _something_ went wrong. :("))
    return largest_clique


SAMPLE_INPUTS = [
    """\
kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 7


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == "co,de,ka,ta"
