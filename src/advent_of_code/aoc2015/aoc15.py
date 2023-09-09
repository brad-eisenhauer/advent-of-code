"""Advent of Code 2015, day 15: https://adventofcode.com/2015/day/15"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Any, Iterable, Iterator, Optional

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            ingredients = [Ingredient.from_str(line.strip()) for line in f]
        result = calc_optimal_recipe(ingredients, 100)
        return result.score()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            ingredients = [Ingredient.from_str(line.strip()) for line in f]
        result = calc_optimal_recipe(ingredients, 100, 500)
        return result.score()


def calc_optimal_recipe(ingredients: list[Ingredient], total_qty: int, target_calorie_count: Optional[int] = None) -> Recipe:
    solution = max(
        r
        for qtys in generate_quantities(len(ingredients), total_qty)
        for r in (Recipe(zip(ingredients, qtys)),)
        if target_calorie_count is None or r.contents()["calories"] == target_calorie_count
    )
    return solution


@dataclass(frozen=True)
class Ingredient:
    name: str
    capacity: int
    durability: int
    flavor: int
    texture: int
    calories: int

    @classmethod
    def from_str(cls, text: str) -> Ingredient:
        name, properties = text.strip().split(": ")
        properties_list = properties.strip().split(", ")
        properties_dict = {
            key: int(val) for prop in properties_list for key, val in (prop.strip().split(" "),)
        }
        return cls(name=name, **properties_dict)


class Recipe:
    def __init__(self, ingredients: Iterable[tuple[Ingredient, int]]):
        self.ingredients = list(ingredients)

    def contents(self) -> dict[str, int]:
        result = defaultdict(int)
        for ingredient, qty in self.ingredients:
            for prop, amt in vars(ingredient).items():
                if isinstance(amt, int):
                    result[prop] += qty * amt
        return result

    def score(self) -> int:
        result = 1
        for prop, amt in self.contents().items():
            if prop != "calories":
                result *= max(0, amt)
        return result

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Recipe):
            raise TypeError(f"Unsupported operand '<' between types 'Recipe' and '{type(other)}'.")
        self_score = self.score()
        other_score = other.score()
        if self_score or other_score:
            return self_score < other_score
        # If both scores are zero we compare by lowest property scores
        self_props = sorted(self.contents().values())
        other_props = sorted(other.contents().values())
        return self_props < other_props

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Recipe) and self.ingredients == other.ingredients


def generate_quantities(ingredient_count: int, total_qty: int) -> Iterator[tuple[int, ...]]:
    if ingredient_count == 1:
        yield (total_qty,)
    else:
        for qty in range(total_qty + 1):
            for qtys in generate_quantities(ingredient_count - 1, total_qty - qty):
                yield (qty, *qtys)


SAMPLE_INPUTS = [
    """\
Butterscotch: capacity -1, durability -2, flavor 6, texture 3, calories 8
Cinnamon: capacity 2, durability 3, flavor -2, texture -1, calories 3
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture
def ingredients():
    return [
        Ingredient("Butterscotch", capacity=-1, durability=-2, flavor=6, texture=3, calories=8),
        Ingredient("Cinnamon", capacity=2, durability=3, flavor=-2, texture=-1, calories=3),
    ]

@pytest.fixture
def recipe(ingredients):
    return Recipe(list(zip(ingredients, [44, 56])))


def test_ingredient_from_str(sample_input, ingredients):
    assert [Ingredient.from_str(line) for line in sample_input] == ingredients


def test_recipe_contents(recipe):
    assert recipe.contents() == {
        "capacity": 68,
        "durability": 80,
        "flavor": 152,
        "texture": 76,
        "calories": 520,
    }

def test_recipe_score(recipe):
    assert recipe.score() == 62842880


def test_recipe_lt(ingredients, recipe):
    recipe_2 = Recipe(list(zip(ingredients, [43, 57])))
    assert recipe_2 < recipe


def test_generate_quantities():
    result_count = sum(1 for _ in generate_quantities(2, 100))
    assert result_count == 101


def test_calc_optimal_recipe(ingredients, recipe):
    assert calc_optimal_recipe(ingredients, 100) == recipe
