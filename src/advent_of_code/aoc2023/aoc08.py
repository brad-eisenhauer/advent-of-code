"""Advent of Code 2023, day 8: https://adventofcode.com/2023/day/8"""

from __future__ import annotations

import re
from functools import reduce
from io import StringIO
from itertools import chain, repeat
from typing import IO, Callable, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.math import least_common_multiple


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(8, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            instructions, node_map = read_map(fp)
        return count_steps_to_goal(instructions, node_map)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            instructions, node_map = read_map(fp)
        return count_steps_to_multi_goal(instructions, node_map)


NodeMap = dict[str, tuple[str, str]]


def read_map(file: IO) -> tuple[str, NodeMap]:
    instructions = file.readline().strip()
    file.readline()
    node_map: dict[str, tuple[str, str]] = {}
    pattern = r"(\w+) = \((\w+), (\w+)\)"
    for line in file:
        match = re.match(pattern, line)
        source, left_dest, right_dest = match.groups()
        node_map[source] = (left_dest, right_dest)
    return instructions, node_map


def count_steps_to_goal(
    instructions: str, node_map: NodeMap, start: str = "AAA", goal: str = "ZZZ"
) -> int:
    _, result = next(find_goals(instructions, node_map, start, lambda n: n == goal))
    return result


def count_steps_to_multi_goal(instructions: str, node_map: NodeMap) -> int:
    start_positions = tuple(key for key in node_map if key.endswith("A"))
    log.debug("%s", start_positions)
    loops: list[tuple[int, int, dict[str, int]]] = [
        find_goal_loop(instructions, node_map, start, lambda n: n.endswith("Z"))
        for start in start_positions
    ]
    log.debug("%s", loops)
    if all(loop[0] == loop[1] and len(loop[2]) == 1 for loop in loops):
        return reduce(least_common_multiple, (loop[0] for loop in loops))
    raise NotImplementedError("I give up. The general case is too hard.")


def find_goals(
    instructions: str, node_map: NodeMap, start: str, predicate: Callable[[str], bool]
) -> Iterator[tuple[str, int]]:
    steps = chain.from_iterable(repeat(instructions))
    step_count = 0
    current_node = start
    while True:
        if predicate(current_node):
            yield current_node, step_count
        match next(steps):
            case "L":
                current_node = node_map[current_node][0]
            case "R":
                current_node = node_map[current_node][1]
        step_count += 1


def find_goal_loop(
    instructions: str, node_map: NodeMap, start: str, predicate: Callable[[str], bool]
) -> tuple[int, int, dict[str, int]]:
    """Find loops in goals.

    Returns
    -------
    int, int, dict[str, int]
        - Steps to start of loop
        - Length of loop
        - Goal/s contained in the loop and the number of steps to each (first encounter).
    """
    goals: dict[str, int] = {}
    for goal, count in find_goals(instructions, node_map, start, predicate):
        if goal in goals:
            loop_start = goals[goal]
            loop_length = count - goals[goal]
            return loop_start, loop_length, goals
        goals[goal] = count
    raise ValueError()


SAMPLE_INPUTS = [
    """\
RL

AAA = (BBB, CCC)
BBB = (DDD, EEE)
CCC = (ZZZ, GGG)
DDD = (DDD, DDD)
EEE = (EEE, EEE)
GGG = (GGG, GGG)
ZZZ = (ZZZ, ZZZ)
""",
    """\
LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)
""",
    """\
LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize(("sample_input", "expected"), [(0, 2), (1, 6)], indirect=["sample_input"])
def test_part_one(solution: AocSolution, sample_input: IO, expected: int):
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize(("sample_input", "expected"), [(2, 6)], indirect=["sample_input"])
def test_part_two(solution: AocSolution, sample_input: IO, expected: int):
    assert solution.solve_part_two(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (
            0,
            (
                "RL",
                {
                    "AAA": ("BBB", "CCC"),
                    "BBB": ("DDD", "EEE"),
                    "CCC": ("ZZZ", "GGG"),
                    "DDD": ("DDD", "DDD"),
                    "EEE": ("EEE", "EEE"),
                    "GGG": ("GGG", "GGG"),
                    "ZZZ": ("ZZZ", "ZZZ"),
                },
            ),
        ),
        (
            1,
            (
                "LLR",
                {
                    "AAA": ("BBB", "BBB"),
                    "BBB": ("AAA", "ZZZ"),
                    "ZZZ": ("ZZZ", "ZZZ"),
                },
            ),
        ),
    ],
    indirect=["sample_input"],
)
def test_read_map(sample_input: IO, expected: tuple[str, dict[str, tuple[str, str]]]):
    assert read_map(sample_input) == expected


@pytest.mark.parametrize(
    ("sample_input", "start", "expected"),
    [
        (0, "AAA", (2, 1, {"ZZZ": 2})),
        (1, "AAA", (6, 1, {"ZZZ": 6})),
        (2, "11A", (2, 2, {"11Z": 2})),
        (2, "22A", (3, 3, {"22Z": 3})),
    ],
    indirect=["sample_input"],
)
def test_find_loop(sample_input: IO, start: str, expected: tuple[int, int, dict[str, int]]):
    instructions, node_map = read_map(sample_input)
    assert find_goal_loop(instructions, node_map, start, lambda s: s.endswith("Z")) == expected
