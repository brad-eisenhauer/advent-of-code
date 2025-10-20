from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            tree = Tree.from_input(f)
        return sum(n.depth for n in tree.contents.values())

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            tree = Tree.from_input(f)
        return tree.calc_transfer_count("YOU", "SAN")


@dataclass(unsafe_hash=True)
class Tree:
    key: str
    parent: Optional[Tree] = None
    branches: list[Tree] = field(default_factory=list, hash=False)

    @classmethod
    def from_input(cls, f: TextIO) -> Tree:
        all_trees = {}

        def get_or_create_tree(key) -> Tree:
            result = all_trees.get(key)
            if result is None:
                result = cls(key)
                all_trees[key] = result
            return result

        for line in f.readlines():
            parent_key, child_key = line.strip().split(")")
            parent = get_or_create_tree(parent_key)
            child = get_or_create_tree(child_key)
            parent.add(child)

        # find root
        t = next(iter(all_trees.values()))
        while t.parent is not None:
            t = t.parent
        return t

    def add(self, child: Tree):
        self.branches.append(child)
        child.parent = self

    @cached_property
    def contents(self) -> dict[str, Tree]:
        result = {self.key: self}
        for branch in self.branches:
            result |= branch.contents
        return result

    @property
    def depth(self):
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    def find_minimum_tree(self, left_key: str, right_key: str) -> Tree:
        contents = self.contents
        if left_key not in contents:
            raise KeyError(left_key)
        if right_key not in contents:
            raise KeyError(right_key)
        t = contents[left_key].parent
        while right_key not in t.contents:
            t = t.parent
        return t

    def calc_transfer_count(self, left_key: str, right_key: str) -> int:
        min_tree = self.find_minimum_tree(left_key, right_key)
        contents = min_tree.contents
        left = contents[left_key]
        right = contents[right_key]
        return (left.parent.depth - min_tree.depth) + (right.parent.depth - min_tree.depth)


SAMPLE_INPUT_1 = """\
COM)B
B)C
C)D
D)E
E)F
B)G
G)H
D)I
E)J
J)K
K)L
"""


@pytest.fixture
def sample_input_1() -> TextIO:
    with StringIO(SAMPLE_INPUT_1) as f:
        yield f


def test_total_orbits(sample_input_1):
    tree = Tree.from_input(sample_input_1)
    result = sum(n.depth for n in tree.contents.values())
    assert result == 42


SAMPLE_INPUT_2 = """\
COM)B
B)C
C)D
D)E
E)F
B)G
G)H
D)I
E)J
J)K
K)L
K)YOU
I)SAN
"""


@pytest.fixture
def sample_input_2() -> TextIO:
    with StringIO(SAMPLE_INPUT_2) as f:
        yield f


def test_find_minimum_tree(sample_input_2):
    tree = Tree.from_input(sample_input_2)
    t = tree.find_minimum_tree("YOU", "SAN")
    assert t.key == "D"


def test_transfer_count(sample_input_2):
    tree = Tree.from_input(sample_input_2)
    result = tree.calc_transfer_count("YOU", "SAN")
    assert result == 4
