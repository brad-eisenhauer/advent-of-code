"""Advent of Code 2016, day 11: https://adventofcode.com/2016/day/11"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from io import StringIO
from itertools import chain, combinations
from typing import IO, Iterator, Literal

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            initial_state = State.read(fp)
        solver = Solver()
        result = solver.find_min_cost_to_goal(initial_state.normalize())
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            initial_state = State.read(fp)
        initial_state = replace(
            initial_state,
            items=initial_state.items
            | {
                (1, Item(type="microship", element="elerium")),
                (1, Item(type="generator", element="elerium")),
                (1, Item(type="microchip", element="dilithium")),
                (1, Item(type="generator", element="dilithium")),
            },
        )
        solver = Solver()
        result = solver.find_min_cost_to_goal(initial_state.normalize())
        return result


@dataclass(frozen=True)
class Item:
    type: Literal["microchip", "generator"]
    element: str


@dataclass(frozen=True)
class State:
    items: frozenset[tuple[int, Item]]
    elevator_floor: int = 1

    @classmethod
    def read(cls, fp: IO) -> State:
        pattern = r"a ([a-z]+)(?: (generator)|-compatible (microchip))"
        items = []
        for i, line in enumerate(fp):
            floor = i + 1
            for match in re.findall(pattern, line):
                item = Item(type=match[1] or match[2], element=match[0])
                items.append((floor, item))
        return cls(items=frozenset(items))

    def normalize(self) -> State:
        element_mapping = {}
        sorted_items = sorted(self.items, key=lambda item: (item[0], item[1].element, item[1].type))
        next_element = "A"
        new_items = []
        for floor, item in sorted_items:
            if item.element not in element_mapping:
                element_mapping[item.element] = next_element
                next_element = chr(ord(next_element) + 1)
            new_items.append((floor, replace(item, element=element_mapping[item.element])))
        return replace(self, items=frozenset(new_items))

    def is_valid(self) -> bool:
        for mc_floor, mc in self.microchips:
            if any(
                g.element == mc.element for g_floor, g in self.generators if g_floor == mc_floor
            ):
                continue
            if any(g_floor == mc_floor for g_floor, _ in self.generators):
                return False
        return True

    @property
    def microchips(self) -> Iterator[Item]:
        return (item for item in self.items if item[1].type == "microchip")

    @property
    def generators(self) -> Iterator[Item]:
        return (item for item in self.items if item[1].type == "generator")

    def floor_is_empty(self, floor: int) -> bool:
        return not any(f == floor for f, _ in self.items)

    def __lt__(self, other: State) -> bool:
        return self.elevator_floor < other.elevator_floor


class Solver(AStar[State]):
    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        # given the items on the current floor (elevator) move 1 or 2 items to another floor;
        # intervening floors must be valid states; skip empty floors if only carrying one item.
        current_items = [item for floor, item in state.items if floor == state.elevator_floor]
        for transporting_items in chain(
            combinations(current_items, 1), combinations(current_items, 2)
        ):
            remaining_items = frozenset(
                item for item in state.items if item[1] not in transporting_items
            )
            for direction in [1, -1]:
                dest_floor = state.elevator_floor + direction
                if dest_floor not in range(1, 5):
                    continue
                next_state = State(
                    items=frozenset(
                        chain(remaining_items, ((dest_floor, item) for item in transporting_items))
                    ),
                    elevator_floor=dest_floor,
                )
                if next_state.is_valid():
                    yield 1, next_state.normalize()

    def is_goal_state(self, state: State) -> bool:
        return all(floor == 4 for floor, _ in state.items)

    def heuristic(self, state: State) -> int:
        result = 0
        for floor, _ in state.items:
            result += 4 - floor
        result -= 4 - state.elevator_floor
        return result


SAMPLE_INPUTS = [
    """\
The first floor contains a hydrogen-compatible microchip and a lithium-compatible microchip.
The second floor contains a hydrogen generator.
The third floor contains a lithium generator.
The fourth floor contains nothing relevant.
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read_state(sample_input):
    expected = State(
        items=frozenset(
            [
                (1, Item(type="microchip", element="hydrogen")),
                (1, Item(type="microchip", element="lithium")),
                (2, Item(type="generator", element="hydrogen")),
                (3, Item(type="generator", element="lithium")),
            ]
        )
    )
    assert State.read(sample_input) == expected


def test_find_shortest_solution(sample_input):
    solver = Solver()
    initial_state = State.read(sample_input)
    assert solver.find_min_cost_to_goal(initial_state) == 11
