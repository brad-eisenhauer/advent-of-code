"""Advent of Code 2022, day 19: https://adventofcode.com/2022/day/19"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from io import StringIO
from time import monotonic
from typing import Collection, Iterator, Mapping

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(19, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            text = f.readlines()
        result = 0
        for i, line in enumerate(text):
            blueprint = Blueprint.parse(line)
            print(f"Calculating for blueprint {blueprint.id} ({i + 1} of {len(text)})...")
            start = monotonic()
            geode_count = calc_geodes_collected(blueprint)
            print(f"Collected {geode_count} geodes in {monotonic() - start:.1f} seconds.")
            result += geode_count * blueprint.id
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            text = f.readlines()[:3]
        result = 1
        for line in text:
            blueprint = Blueprint.parse(line)
            print(f"Calculating for blueprint {blueprint.id}...")
            start = monotonic()
            geode_count = calc_geodes_collected(blueprint, time=32)
            print(f"Collected {geode_count} geodes in {monotonic() - start:.1f} seconds.")
            result *= geode_count
        return result


@dataclass
class Recipe:
    produces: str
    materials: dict[str, int] = field(default_factory=dict)

    def can_make_with(self, inv_materials: Mapping[str, int]) -> bool:
        return all(
            qty_required <= inv_materials.get(resource, 0)
            for resource, qty_required in self.materials.items()
        )


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
            recipes.append(recipe)
            recipe_part = recipe_part[match_.end() :]
        return Blueprint(int(id_part.split()[1]), recipes)


@dataclass(unsafe_hash=True)
class Inventory:
    materials: tuple[tuple[str, int], ...]
    robots: tuple[str, ...]

    def __lt__(self, other):
        if self.robots != other.robots:
            return self.robots < other.robots
        return self.materials < other.materials

    def calc_next_materials(self) -> Mapping[str, int]:
        mats = defaultdict(int, self.materials)
        for name in self.robots:
            mats[name] += 1
        return mats

    def get_materials_as_dict(self) -> Mapping[str, int]:
        return dict(self.materials)

    def count_robots(self) -> Mapping[str, int]:
        return Counter(self.robots)


@dataclass(frozen=True)
class State:
    inventory: Inventory
    time_remaining: int

    def __lt__(self, other):
        if self.time_remaining != other.time_remaining:
            return self.time_remaining < other.time_remaining
        return self.inventory < other.inventory


class Navigator(AStar[State]):
    def __init__(self, blueprint: Blueprint, target_resource: str = "geode"):
        self._blueprint = blueprint
        self._target_resource = target_resource
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
        post_time = state.time_remaining - 1  # time remaining at the end of succeeding minute
        robot_counts = Counter(state.inventory.robots)

        for recipe in self._blueprint.recipes:
            # If we're already gathering as much of a resource each minute as we could ever
            # consume, it will never be our limiting resource. There's no point in building
            # robots to gather more.
            if (
                recipe.produces in self._max_consumption
                and robot_counts.get(recipe.produces, 0) >= self._max_consumption[recipe.produces]
            ):
                continue
            # Check if we have the resources to build this robot.
            if not recipe.can_make_with(state.inventory.get_materials_as_dict()):
                continue
            # Add resources that will be gathered.
            materials = state.inventory.calc_next_materials()
            # Remove resources to build the next robot.
            for resource, qty in recipe.materials.items():
                materials[resource] -= qty
            next_state = State(
                inventory=Inventory(
                    materials=tuple(materials.items()),
                    robots=(recipe.produces, *state.inventory.robots),
                ),
                time_remaining=post_time,
            )
            # Without resource constraints, we could build one robot to gather our target
            # resource every minute. Each such robot would eventually gather a quantity of the
            # resource equal to `post_time`. The cost of building something else, or building
            # nothing, is therefore equal to `post_time`.
            cost = 0 if recipe.produces == self._target_resource else post_time
            yield cost, next_state

        # produce nothing; just gather materials
        materials = state.inventory.calc_next_materials()
        next_inv = Inventory(materials=tuple(materials.items()), robots=state.inventory.robots)
        yield post_time, State(next_inv, post_time)

    def heuristic(self, state: State) -> int:
        """Conservative minimal cost to end"""
        target_recipe = next(
            r for r in self._blueprint.recipes if r.produces == self._target_resource
        )
        if target_recipe.can_make_with(state.inventory.get_materials_as_dict()):
            return 0
        next_mats = state.inventory.calc_next_materials()
        if target_recipe.can_make_with(next_mats):
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
    return final_state.inventory.get_materials_as_dict().get("geode", 0)


SAMPLE_INPUTS = [
    """\
Blueprint 1: Each ore robot costs 4 ore. Each clay robot costs 2 ore. Each obsidian robot costs 3 ore and 14 clay. Each geode robot costs 2 ore and 7 obsidian.
Blueprint 2: Each ore robot costs 2 ore. Each clay robot costs 3 ore. Each obsidian robot costs 3 ore and 8 clay. Each geode robot costs 3 ore and 12 obsidian.
Blueprint 3: Each ore robot costs 1 ore. Each clay robot costs 1 ore. Each obsidian robot costs 1 ore and 1 clay. Each geode robot costs 0 ore and 0 obsidian.
Blueprint 4: Each ore robot costs 1 ore. Each clay robot costs 1 ore. Each obsidian robot costs 1 ore and 1 clay. Each geode robot costs 1 ore and 1 obsidian.
""",  # noqa: E501
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture
def blueprint(sample_input, request):
    line = sample_input.readlines()[request.param]
    return Blueprint.parse(line)


def test_blueprint_parse(sample_input):
    assert Blueprint.parse(sample_input.readline()) == Blueprint(
        id=1,
        recipes=[
            Recipe("ore", {"ore": 4}),
            Recipe("clay", {"ore": 2}),
            Recipe("obsidian", {"ore": 3, "clay": 14}),
            Recipe("geode", {"ore": 2, "obsidian": 7}),
        ],
    )


@pytest.mark.parametrize(
    ("blueprint", "expected"), [(0, 9), (1, 12), (2, 276), (3, 171)], indirect=["blueprint"]
)
def test_calc_geodes_collected(blueprint, expected):
    assert calc_geodes_collected(blueprint) == expected


@pytest.mark.parametrize("blueprint", list(range(4)), indirect=True)
def test_cost_plus_geodes_is_constant(blueprint):
    allotted_time = 24
    nav = Navigator(blueprint)
    initial_state = State(
        inventory=Inventory(materials=(), robots=("ore",)), time_remaining=allotted_time
    )

    final_state, costs, _ = nav._find_min_cost_path(initial_state)

    expected_total = allotted_time * (allotted_time - 1) // 2
    assert (
        final_state.inventory.get_materials_as_dict().get("geode") + costs[final_state]
        == expected_total
    )
