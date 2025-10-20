"""Advent of Code 2022, day 23: https://adventofcode.com/2022/day/23"""

from __future__ import annotations

from collections import Counter
from io import StringIO
from itertools import count
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(23, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            elves = read(f)
        elves = run(elves, 10)
        return calc_open_space(elves)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            elves = read(f)
        result, _ = run_to_stasis(elves)
        return result


Vector = tuple[int, ...]


def read(f: TextIO) -> list[Vector]:
    return [
        (col, row)
        for row, line in enumerate(f)
        for col, char in enumerate(line.rstrip())
        if char == "#"
    ]


MOVE_VECTORS: list[tuple[int, ...]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def propose_moves(elves: list[Vector], move_index: int) -> Iterator[Vector]:
    occupied = set(elves)
    move_vectors = [MOVE_VECTORS[(move_index + i) % 4] for i in range(4)]
    for elf in elves:
        has_neighbors: list[bool] = []
        for move in move_vectors:
            for offset in (-1, 0, 1):
                neighbor_vector = (offset, move[1]) if move[0] == 0 else (move[0], offset)
                if tuple(a + b for a, b in zip(elf, neighbor_vector)) in occupied:
                    has_neighbors.append(True)
                    break
            else:
                has_neighbors.append(False)

        if not any(has_neighbors):
            yield elf
        else:
            for move, neighbors in zip(move_vectors, has_neighbors):
                if not neighbors:
                    yield tuple(a + b for a, b in zip(elf, move))
                    break
            else:
                yield elf


def make_moves(elves: list[Vector], proposed_moves: list[Vector]) -> list[Vector]:
    proposed_counts = Counter(proposed_moves)
    return [new if proposed_counts[new] < 2 else old for new, old in zip(proposed_moves, elves)]


def run(elves: list[Vector], n: int, move_index: int = 0) -> list[Vector]:
    for i in range(n):
        moves = list(propose_moves(elves, (move_index + i) % 4))
        elves = make_moves(elves, moves)
    return elves


def calc_open_space(elves: list[Vector]) -> int:
    min_x, min_y = (min(ns) for ns in zip(*elves))
    max_x, max_y = (max(ns) for ns in zip(*elves))
    area = (max_x - min_x + 1) * (max_y - min_y + 1)
    return area - len(elves)


def run_to_stasis(elves: list[Vector]) -> tuple[int, list[Vector]]:
    for i in count(0):
        new_elves = run(elves, 1, i)
        if new_elves == elves:
            return i + 1, new_elves
        elves = new_elves
    raise ValueError()


SAMPLE_INPUTS = [
    """\
....#..
..###.#
#...#.#
.#...##
#.###..
##.#.##
.#..#..
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read(sample_input):
    expected = [
        # fmt: off
        (4, 0),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (0, 2),
        (4, 2),
        (6, 2),
        (1, 3),
        (5, 3),
        (6, 3),
        (0, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (0, 5),
        (1, 5),
        (3, 5),
        (5, 5),
        (6, 5),
        (1, 6),
        (4, 6),
        # fmt: on
    ]
    assert read(sample_input) == expected


def test_propose_moves(sample_input):
    expected = [
        # fmt: off
        (4, -1),
        (2, 0),
        (3, 1),
        (5, 1),
        (6, 0),
        (0, 1),
        (4, 2),
        (7, 2),
        (1, 3),
        (5, 3),
        (6, 4),
        (-1, 4),
        (2, 4),
        (3, 3),
        (4, 4),
        (-1, 5),
        (1, 5),
        (3, 5),
        (5, 5),
        (6, 4),
        (1, 7),
        (4, 7),
        # fmt: on
    ]
    elves = read(sample_input)
    result = list(propose_moves(elves, 0))
    assert result == expected


def test_make_moves(sample_input):
    expected = [
        # fmt: off
        (4, -1),
        (2, 0),
        (3, 1),
        (5, 1),
        (6, 0),
        (0, 1),
        (4, 2),
        (7, 2),
        (1, 3),
        (5, 3),
        (6, 3),
        (-1, 4),
        (2, 4),
        (3, 3),
        (4, 4),
        (-1, 5),
        (1, 5),
        (3, 5),
        (5, 5),
        (6, 5),
        (1, 7),
        (4, 7),
        # fmt: on
    ]
    elves = read(sample_input)
    moves = list(propose_moves(elves, 0))
    new_elves = make_moves(elves, moves)
    assert new_elves == expected


def test_count_open_space(sample_input):
    elves = read(sample_input)
    elves = run(elves, 10)
    assert calc_open_space(elves) == 110


def test_run_to_stasis(sample_input):
    elves = read(sample_input)
    result, _ = run_to_stasis(elves)
    assert result == 20
