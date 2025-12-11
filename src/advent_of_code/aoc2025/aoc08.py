"""Advent of Code 2025, day 8: https://adventofcode.com/2025/day/8"""

from __future__ import annotations

import logging
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from heapq import heapify, heappop
from io import StringIO
from itertools import combinations, islice
from typing import IO, Iterable, Iterator, Optional, Self, TypeAlias

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)

JunctionBox: TypeAlias = tuple[int, int, int]
Connection: TypeAlias = tuple[JunctionBox, JunctionBox]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, n_connections=1000) -> int:
        with input_file or self.open_input() as fp:
            ds: DisjointSet[JunctionBox] = DisjointSet.from_elements(eval(line) for line in fp)
        for left, right in islice(generate_shortest_connections(ds), n_connections):
            ds.union(left, right)
        circuit_sizes = ds.calc_set_sizes()
        largest_sizes = sorted(circuit_sizes.values(), reverse=True)[:3]
        return reduce(operator.mul, largest_sizes)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ds: DisjointSet[JunctionBox] = DisjointSet.from_elements(eval(line) for line in fp)
        all_roots = set(ds.parents)
        for left, right in generate_shortest_connections(ds):
            if (merged_root := ds.union(left, right)) is not None:
                all_roots.remove(merged_root)
            if len(all_roots) == 1:
                return left[0] * right[0]


@dataclass
class DisjointSet[T]:
    parents: dict[T, T] = field(default_factory=dict)
    ranks: dict[T, int] = field(default_factory=dict)

    @classmethod
    def from_elements(cls, items: Iterable[T]) -> Self:
        parents = {item: item for item in items}
        ranks = dict.fromkeys(parents, 0)
        return cls(parents=parents, ranks=ranks)

    def find(self, item: T) -> T:
        if item not in self.parents:
            self.parents[item] = item
            self.ranks[item] = 0
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, left: T, right: T) -> T | None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return None

        if self.ranks[root_left] > self.ranks[root_right]:
            self.parents[root_right] = root_left
            return root_right
        elif self.ranks[root_left] < self.ranks[root_right]:
            self.parents[root_left] = root_right
            return root_left
        else:
            self.parents[root_right] = root_left
            self.ranks[root_left] += 1
            return root_right

    def calc_set_sizes(self) -> dict[T, int]:
        result = defaultdict(int)
        for item in self.parents:
            result[self.find(item)] += 1
        return result


def generate_shortest_connections(boxes: DisjointSet[JunctionBox]) -> Iterator[Connection]:
    connections = combinations(boxes.parents, 2)
    connection_heap = [(calc_distance_sq(*pair), pair) for pair in connections]
    heapify(connection_heap)
    while connection_heap:
        yield heappop(connection_heap)[1]


def calc_distance_sq(left: JunctionBox, right: JunctionBox) -> int:
    return sum((a - b) ** 2 for a, b in zip(left, right))


SAMPLE_INPUTS = [
    """\
162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input, n_connections=10) == 40


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 25272
