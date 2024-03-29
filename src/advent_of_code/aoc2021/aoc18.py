""" Advent of Code 2021, Day 18: https://adventofcode.com/2021/day/18 """
from __future__ import annotations

import json
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from itertools import permutations
from typing import Iterable, Iterator, Optional, Sequence, TextIO, Union

import pytest

from advent_of_code.base import Solution

MAX_ALLOWABLE_DEPTH = 4
MAX_ALLOWABLE_LEAF = 9


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(18, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            result = reduce(operator.add, parse_input(f))
        return result.magnitude

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            sf_numbers = tuple(parse_input(f))
        return find_largest_sum(sf_numbers)


@dataclass
class ExplodeResult:
    left_remainder: Optional[int]
    result: Tree
    right_remainder: Optional[int]


class Tree(ABC):
    @classmethod
    def from_iterable(cls, source: Iterable[Union[Iterable, int]]) -> Tree:
        def build_child(item: Union[Iterable, int]) -> Tree:
            if isinstance(item, Iterable):
                return Tree.from_iterable(item)
            return Leaf(item)

        left, right = source
        return Node(build_child(left), build_child(right))

    @property
    @abstractmethod
    def magnitude(self) -> int:
        ...

    def __add__(self, other: Tree) -> Tree:
        return Node(self, other).reduce()

    def reduce(self) -> Tree:
        result = self
        while True:
            explode_result = result.explode()
            if explode_result.result is not result:
                result = explode_result.result
            # .explode() performs all explosions, so we can immediately start searching
            # for splits.

            split_result = result.split()
            if split_result is not result:
                result = split_result
                continue  # If a split occurs, we need to go back a look for explosions.

            return result

    @abstractmethod
    def explode(self, depth: int = 1) -> ExplodeResult:
        ...

    @abstractmethod
    def add_left(self, num: Optional[int]) -> Tree:
        ...

    @abstractmethod
    def add_right(self, num: Optional[int]) -> Tree:
        ...

    @abstractmethod
    def split(self) -> Tree:
        ...


@dataclass(frozen=True)
class Leaf(Tree):
    value: int

    def __repr__(self):
        return f"{self.value}"

    @property
    def magnitude(self) -> int:
        return self.value

    def explode(self, depth: int = 1) -> ExplodeResult:
        return ExplodeResult(None, self, None)

    def split(self) -> Tree:
        if self.value > MAX_ALLOWABLE_LEAF:
            return Node(Leaf(self.value // 2), Leaf((self.value + 1) // 2))
        return self

    def add_left(self, num: Optional[int]) -> Tree:
        if num:
            return Leaf(self.value + num)
        return self

    def add_right(self, num: Optional[int]) -> Tree:
        if num:
            return Leaf(self.value + num)
        return self


@dataclass(frozen=True)
class Node(Tree):
    left: Tree
    right: Tree

    def __repr__(self):
        return f"[{self.left!r}, {self.right!r}]"

    @property
    def magnitude(self) -> int:
        return 3 * self.left.magnitude + 2 * self.right.magnitude

    def explode(self, depth: int = 1) -> ExplodeResult:
        """Perform all explosions, from left to right"""
        if depth > MAX_ALLOWABLE_DEPTH:
            return ExplodeResult(self.left.magnitude, Leaf(0), self.right.magnitude)

        left_result = self.left
        right_result = self.right
        left_remainder: Optional[int] = None
        right_remainder: Optional[int] = None

        explode_result = left_result.explode(depth + 1)
        if explode_result.result is not left_result:
            left_result = explode_result.result
            left_remainder = explode_result.left_remainder
            right_result = right_result.add_left(explode_result.right_remainder)

        explode_result = right_result.explode(depth + 1)
        if explode_result.result is not right_result:
            right_result = explode_result.result
            right_remainder = explode_result.right_remainder
            left_result = left_result.add_right(explode_result.left_remainder)

        if left_result is not self.left or right_result is not self.right:
            return ExplodeResult(left_remainder, Node(left_result, right_result), right_remainder)

        return ExplodeResult(None, self, None)

    def add_right(self, num: Optional[int]) -> Tree:
        if num:
            return Node(self.left, self.right.add_right(num))
        return self

    def add_left(self, num: Optional[int]) -> Tree:
        if num:
            return Node(self.left.add_left(num), self.right)
        return self

    def split(self) -> Tree:
        left_split = self.left.split()
        if left_split is not self.left:
            return Node(left_split, self.right)

        right_split = self.right.split()
        if right_split is not self.right:
            return Node(self.left, right_split)

        return self


def parse_input(fp: TextIO) -> Iterator[Tree]:
    for line in fp:
        sf_num = json.loads(line.strip())
        yield Tree.from_iterable(sf_num)


def find_largest_sum(sf_numbers: Sequence[Tree]) -> int:
    return max(
        (left + right).magnitude for left, right in permutations(sf_numbers, 2) if left is not right
    )


SAMPLE_INPUT = """\
[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
[[[5,[2,8]],4],[5,[[9,9],0]]]
[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
[[[[5,4],[7,7]],8],[[8,3],8]]
[[9,3],[[9,9],[6,[4,9]]]]
[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]
"""


@pytest.fixture()
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_calc_sum(sample_input):
    numbers = parse_input(sample_input)
    result = next(numbers)
    for addend in numbers:
        result += addend
    assert result == Tree.from_iterable(
        [[[[6, 6], [7, 6]], [[7, 7], [7, 0]]], [[[7, 7], [7, 7]], [[7, 8], [9, 9]]]]
    )


@pytest.mark.parametrize(
    ("sf_num", "expected"),
    (
        ([[1, 2], [[3, 4], 5]], 143),
        ([[[[0, 7], 4], [[7, 8], [6, 0]]], [8, 1]], 1384),
        ([[[[1, 1], [2, 2]], [3, 3]], [4, 4]], 445),
        ([[[[3, 0], [5, 3]], [4, 4]], [5, 5]], 791),
        ([[[[5, 0], [7, 4]], [5, 5]], [6, 6]], 1137),
        ([[[[8, 7], [7, 7]], [[8, 6], [7, 7]]], [[[0, 7], [6, 6]], [8, 7]]], 3488),
    ),
)
def test_magnitude(sf_num, expected):
    assert Tree.from_iterable(sf_num).magnitude == expected


@pytest.mark.parametrize(
    ("sf_num", "expected"),
    (
        ([[[[[9, 8], 1], 2], 3], 4], [[[[0, 9], 2], 3], 4]),
        ([7, [6, [5, [4, [3, 2]]]]], [7, [6, [5, [7, 0]]]]),
        ([[6, [5, [4, [3, 2]]]], 1], [[6, [5, [7, 0]]], 3]),
        (
            [[3, [2, [1, [7, 3]]]], [6, [5, [4, [3, 2]]]]],
            [[3, [2, [8, 0]]], [9, [5, [7, 0]]]],
        ),
        (
            [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]],
            [[3, [2, [8, 0]]], [9, [5, [7, 0]]]],
        ),
    ),
)
def test_explode(sf_num, expected):
    result = Tree.from_iterable(sf_num).explode()
    assert result.result == Tree.from_iterable(expected)


@pytest.mark.parametrize(
    ("sf_num", "expected"),
    (
        (
            [[[[0, 7], 4], [15, [0, 13]]], [1, 1]],
            [[[[0, 7], 4], [[7, 8], [0, 13]]], [1, 1]],
        ),
        (
            [[[[0, 7], 4], [[7, 8], [0, 13]]], [1, 1]],
            [[[[0, 7], 4], [[7, 8], [0, [6, 7]]]], [1, 1]],
        ),
    ),
)
def test_split(sf_num, expected):
    result = Tree.from_iterable(sf_num).split()
    assert result == Tree.from_iterable(expected)
