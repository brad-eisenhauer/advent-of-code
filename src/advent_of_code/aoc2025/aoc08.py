"""Advent of Code 2025, day 8: https://adventofcode.com/2025/day/8"""

from __future__ import annotations

from functools import reduce
import logging
from io import StringIO
from itertools import combinations
import operator
from typing import IO, Collection, Iterator, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, n_connections=1000) -> int:
        with input_file or self.open_input() as fp:
            boxes = list(read_boxes(fp))
        connections = make_shortest_connections(boxes, n_connections)
        circuits = create_circuits(boxes, connections)
        largest = sorted((len(c) for c in circuits), reverse=True)[:3]
        return reduce(operator.mul, largest)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            boxes = list(read_boxes(fp))
        *_, (last_left, last_right) = connect_until_single_circuit(boxes)
        return last_left[0] * last_right[0]


JunctionBox: TypeAlias = tuple[int, int, int]
Connection: TypeAlias = tuple[JunctionBox, JunctionBox]


def calc_distance_sq(left: JunctionBox, right: JunctionBox) -> int:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def read_boxes(reader: IO) -> Iterator[JunctionBox]:
    for line in reader:
        yield eval(line.strip())


def make_shortest_connections(boxes: Collection[JunctionBox], n: int = 1000) -> list[Connection]:
    box_pairs = combinations(boxes, 2)
    shortest_pairs = sorted(box_pairs, key=lambda pair: calc_distance_sq(*pair))
    return shortest_pairs[:n]


def create_circuits(boxes: list[JunctionBox], connections: list[Connection]) -> list[set[JunctionBox]]:
    circuit_map = {box: {box} for box in boxes}
    for left, right in connections:
        if circuit_map[left] is circuit_map[right]:
            continue
        circuit_map[left] |= circuit_map[right]
        for box in circuit_map[left]:
            circuit_map[box] = circuit_map[left]
    # filter for unique values
    result = []
    for circuit in circuit_map.values():
        if circuit not in result:
            result.append(circuit)
    return result


def connect_until_single_circuit(boxes: Collection[JunctionBox]) -> list[Connection]:
    box_pairs = combinations(boxes, 2)
    shortest_pairs = sorted(box_pairs, key=lambda pair: calc_distance_sq(*pair))
    circuit_map = {box: {box} for box in boxes}
    # We'll need at least n-1 connection to join n boxes.
    result = shortest_pairs[:len(boxes) - 1]
    for left, right in shortest_pairs[:len(boxes) - 1]:
        if circuit_map[left] is circuit_map[right]:
            continue
        circuit_map[left] |= circuit_map[right]
        for box in circuit_map[left]:
            circuit_map[box] = circuit_map[left]
    if len(next(iter(circuit_map.values()))) == len(boxes):
        return result
    # Make additional connections until one circuit contains all of the boxes
    for pair in shortest_pairs[len(boxes) - 1:]:
        left, right = pair
        if circuit_map[left] is circuit_map[right]:
            continue
        circuit_map[left] |= circuit_map[right]
        result.append(pair)
        if len(circuit_map[left]) == len(boxes):
            return result
        for box in circuit_map[left]:
            circuit_map[box] = circuit_map[left]
    raise ValueError("no solution found")



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


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input, n_connections=10) == 40


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 25272
