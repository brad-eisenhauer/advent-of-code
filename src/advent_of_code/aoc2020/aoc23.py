"""Advent of Code 2020, day 23: https://adventofcode.com/2020/day/23"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(23, 2020, **kwargs)

    def solve_part_one(self) -> str:
        with self.open_input() as f:
            start, lookup = read_ring(f)
        _ = perform_trick(start, lookup, 100)
        node_1 = lookup[1]
        return "".join(str(n) for n in node_1.get_ring()[1:])

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            start, lookup = read_ring(f, 1_000_000)
        _ = perform_trick(start, lookup, 10_000_000)
        node_1 = lookup[1]
        return node_1.next.id * node_1.next.next.id


@dataclass
class Node:
    id: int
    next: Optional[Node] = None
    previous: Optional[Node] = None
    in_loop: bool = False

    def __eq__(self, other):
        if self.id != other.id:
            return False
        if (self.next is None) != (other.next is None):
            return False
        if (self.previous is None) != (other.previous is None):
            return False
        if self.next is not None and self.next.id != other.next.id:
            return False
        if self.previous is not None and self.previous.id != other.previous.id:
            return False
        return True

    def __repr__(self):
        return f"Node({self.id})"

    def insert(self, node: Node):
        old_next = self.next
        self.next = node
        node.previous = self
        # if node has successors, insert them as well
        last = node
        while True:
            last.in_loop = True
            if (successor := last.next) is None:
                break
            last = successor
        last.next = old_next
        if old_next is not None:
            old_next.previous = last

    def cut(self, n: int) -> Node:
        head = self.next
        end = head
        try:
            for _ in range(n):
                end.in_loop = False
                end = end.next
        except TypeError as e:
            raise ValueError("Not enough links") from e
        head.previous = None
        end.previous.next = None
        self.next = end
        end.previous = self
        return head

    def get_ring(self) -> list[int]:
        result = [self.id]
        node = self.next
        while node is not None and node is not self:
            result.append(node.id)
            node = node.next
        return result


def read_ring(f: TextIO, ring_size: int = 9) -> tuple[Node, dict[int, Node]]:
    prev: Optional[Node] = None
    start: Optional[Node] = None
    lookup: dict[int, Node] = {}
    for char in f.readline().strip():
        node = Node(int(char), previous=prev, in_loop=True)
        lookup[node.id] = node
        if start is None:
            start = node
        if prev is not None:
            prev.next = node
        prev = node
    while len(lookup) < ring_size:
        node = Node(len(lookup) + 1, previous=prev, in_loop=True)
        lookup[node.id] = node
        prev.next = node
        prev = node
    node.next = start
    start.previous = node
    return start, lookup


def perform_trick(initial_ring: Node, lookup: dict[int, Node], steps: int) -> Node:
    ring_length = len(initial_ring.get_ring())
    current = initial_ring
    for _ in range(steps):
        section = current.cut(3)
        dest_id = current.id
        while True:
            dest_id -= 1
            if dest_id < 1:
                dest_id = ring_length
            if lookup[dest_id].in_loop:
                break
        dest = lookup[dest_id]
        dest.insert(section)
        current = current.next
    return current


SAMPLE_INPUTS = [
    """\
389125467
"""
]


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture
def initial_ring(sample_input: TextIO, request) -> tuple[Node, dict[int, Node]]:
    return read_ring(sample_input, request.param)


@pytest.mark.parametrize("initial_ring", [9], indirect=True)
def test_read_ring(initial_ring):
    expected_lookup = {n: Node(n, in_loop=True) for n in range(1, 10)}

    def link(a, b):
        expected_lookup[a].next = expected_lookup[b]
        expected_lookup[b].previous = expected_lookup[a]

    order = [3, 8, 9, 1, 2, 5, 4, 6, 7, 3]
    for a, b in zip(order, order[1:]):
        link(a, b)

    start, lookup = initial_ring
    assert lookup == expected_lookup
    assert start.get_ring() == order[:-1]


@pytest.mark.parametrize("initial_ring", [9], indirect=True)
def test_cut(initial_ring):
    start, _ = initial_ring
    result = start.cut(3)
    assert result.get_ring() == [8, 9, 1]
    assert start.get_ring() == [3, 2, 5, 4, 6, 7]


@pytest.mark.parametrize("initial_ring", [9], indirect=True)
def test_append(initial_ring):
    start, lookup = initial_ring
    section = start.cut(3)
    lookup[2].insert(section)
    assert lookup[2].get_ring() == [2, 8, 9, 1, 5, 4, 6, 7, 3]


@pytest.mark.parametrize("initial_ring", [9], indirect=True)
def test_perform_trick(initial_ring):
    start, lookup = initial_ring
    result = perform_trick(start, lookup, 10)
    assert result.get_ring() == [8, 3, 7, 4, 1, 9, 2, 6, 5]


@pytest.mark.parametrize("initial_ring", [1_000_000], indirect=True)
def test_part_two(initial_ring):
    start, lookup = initial_ring
    _ = perform_trick(start, lookup, 10_000_000)
    node_1 = lookup[1]
    assert node_1.next.id * node_1.next.next.id == 149245887792
