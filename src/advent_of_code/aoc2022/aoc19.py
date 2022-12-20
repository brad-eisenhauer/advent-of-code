"""Advent of Code 2022, day 19: https://adventofcode.com/2022/day/19"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from io import StringIO
from time import monotonic
from typing import Collection, Iterator

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(19, 2022, **kwargs)

    def solve_part_one(self) -> int:
        max_geode_count = 0
        max_geode_blueprint = None
        with self.open_input() as f:
            text = f.readlines()
        for i, line in enumerate(text):
            blueprint = Blueprint.parse(line)
            print(f"Calculating for blueprint {blueprint.id} ({i + 1} of {len(text)})... ", end="")
            start = monotonic()
            geode_count = calc_geodes_collected(blueprint)
            print(f"Collected {geode_count} geodes in {monotonic() - start:.1f} seconds.")
            if geode_count > max_geode_count:
                max_geode_count = geode_count
                max_geode_blueprint = blueprint
        return max_geode_count * max_geode_blueprint.id


@dataclass
class Recipe:
    produces: str
    materials: dict[str, int] = field(default_factory=dict)

    def can_make_with(self, inventory: Inventory) -> bool:
        inv_materials = dict(inventory.materials)
        for resource, qty_on_hand in self.materials.items():
            if inv_materials.get(resource, 0) < qty_on_hand:
                return False
        return True


@dataclass
class Blueprint:
    id: int
    recipes: Collection[Recipe]

    @classmethod
    def parse(cls, line: str) -> Blueprint:
        id_part, recipe_part = line.split(":")
        recipe_pattern = re.compile(r" Each (\w+) robot costs (\d+) (\w+)(?: and (\d+) (\w+))?\.")
        recipes = []
        while (match_ := recipe_pattern.search(recipe_part)) is not None:
            product, *materials = match_.groups()
            recipe = Recipe(produces=product)
            while materials and materials[0] is not None:
                recipe.materials[materials[1]] = int(materials[0])
                materials = materials[2:]
            recipes.insert(0, recipe)
            recipe_part = recipe_part[match_.end() :]
        return Blueprint(int(id_part.split()[1]), recipes)


@dataclass(frozen=True)
class Inventory:
    materials: tuple[tuple[str, int], ...]
    robots: tuple[str, ...]

    def __lt__(self, other):
        if self.robots != other.robots:
            return self.robots < other.robots
        return self.materials < other.materials

    def calc_next_materials(self) -> dict[str, int]:
        mats = defaultdict(int, self.materials)
        for name in self.robots:
            mats[name] += 1
        return mats


@dataclass(frozen=True)
class State:
    inventory: Inventory
    time_remaining: int

    def __lt__(self, other):
        if self.time_remaining != other.time_remaining:
            return self.time_remaining < other.time_remaining
        return self.inventory < other.inventory


class Navigator(AStar[State]):
    def __init__(
        self, blueprint: Blueprint, target_product: str = "geode"
    ):
        self._blueprint = blueprint
        self._target_product = target_product
        self._max_consumption = reduce(
            lambda acc, r: {
                m: max(r.materials.get(m, 0), acc.get(m, 0)) for m in set(acc) | set(r.materials)
            },
            blueprint.recipes,
            defaultdict(int),
        )

    def is_goal_state(self, state: State) -> bool:
        return state.time_remaining == 0

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        time_remaining = state.time_remaining - 1
        robot_counts = Counter(state.inventory.robots)

        for recipe in self._blueprint.recipes:
            if (
                recipe.produces in self._max_consumption
                and robot_counts.get(recipe.produces, 0) >= self._max_consumption[recipe.produces]
            ):
                continue
            if not recipe.can_make_with(state.inventory):
                continue
            cost = 0 if recipe.produces == self._target_product else time_remaining
            materials = state.inventory.calc_next_materials()
            for resource, qty in recipe.materials.items():
                materials[resource] -= qty
            next_inv = Inventory(
                materials=tuple(materials.items()),
                robots=(recipe.produces, *state.inventory.robots),
            )
            next_state = State(inventory=next_inv, time_remaining=time_remaining)
            yield cost, next_state

        # produce nothing; just gather materials
        materials = state.inventory.calc_next_materials()
        next_inv = Inventory(materials=tuple(materials.items()), robots=state.inventory.robots)
        yield time_remaining, State(next_inv, time_remaining)

    def heuristic(self, state: State) -> int:
        for target_recipe in self._blueprint.recipes:
            if target_recipe.produces == self._target_product:
                break
        if target_recipe.can_make_with(state.inventory):
            return 0
        materials = state.inventory.calc_next_materials()
        next_inv = Inventory(materials=tuple(materials.items()), robots=state.inventory.robots)
        if target_recipe.can_make_with(next_inv):
            return max(0, state.time_remaining - 1)
        return max(0, 2 * state.time_remaining - 3)


def calc_geodes_collected(blueprint: Blueprint, time: int = 24) -> int:
    inventory = Inventory(materials=(), robots=("ore",))
    initial_state = State(inventory, time_remaining=time)
    geode_nav = Navigator(blueprint)
    if log.isEnabledFor(logging.DEBUG):
        for final_state, _ in geode_nav.find_min_cost_path(initial_state):
            log.debug("%d: %s", final_state.time_remaining, final_state.inventory.robots)
    else:
        final_state, _, _ = geode_nav._find_min_cost_path(initial_state)
    return dict(final_state.inventory.materials).get("geode", 0)


SAMPLE_INPUTS = [
    """\
Blueprint 1: Each ore robot costs 4 ore. Each clay robot costs 2 ore. Each obsidian robot costs 3 ore and 14 clay. Each geode robot costs 2 ore and 7 obsidian.
Blueprint 2: Each ore robot costs 2 ore. Each clay robot costs 3 ore. Each obsidian robot costs 3 ore and 8 clay. Each geode robot costs 3 ore and 12 obsidian.
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_blueprint_parse(sample_input):
    assert Blueprint.parse(sample_input.readline()) == Blueprint(
        id=1,
        recipes=[
            Recipe("geode", {"ore": 2, "obsidian": 7}),
            Recipe("obsidian", {"ore": 3, "clay": 14}),
            Recipe("clay", {"ore": 2}),
            Recipe("ore", {"ore": 4}),
        ],
    )


@pytest.mark.parametrize(("blueprint_index", "expected"), [(0, 9), (1, 12)])
def test_calc_geodes_collected(sample_input, blueprint_index, expected):
    line = sample_input.readlines()[blueprint_index]
    blueprint = Blueprint.parse(line)
    assert calc_geodes_collected(blueprint) == expected
