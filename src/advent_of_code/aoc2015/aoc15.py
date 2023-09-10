"""Advent of Code 2015, day 15: https://adventofcode.com/2015/day/15"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from io import StringIO
from itertools import permutations
from typing import Any, Callable, Iterable, Iterator, Optional

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            ingredients = [Ingredient.from_str(line.strip()) for line in f]
        result = find_optimal_recipe(ingredients, 100)
        log.debug(result)
        return result.score()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            ingredients = [Ingredient.from_str(line.strip()) for line in f]
        result = find_optimal_recipe_calorie_constrained(ingredients, 100, 500)
        log.debug(result)
        return result.score()


def calc_optimal_recipe(
    ingredients: list[Ingredient], total_qty: int, target_calorie_count: Optional[int] = None
) -> Recipe:
    all_recipes = (
        Recipe(zip(ingredients, qtys)) for qtys in generate_quantities(len(ingredients), total_qty)
    )
    candidate_recipes = (
        all_recipes
        if target_calorie_count is None
        else (r for r in all_recipes if r.contents()["calories"] == target_calorie_count)
    )
    return max(candidate_recipes, key=lambda r: r.score())


def find_optimal_recipe(ingredients: list[Ingredient], total_qty: int) -> Recipe:
    best_recipe = Recipe.equal_parts(ingredients, total_qty)
    best_score = best_recipe.score()
    while True:
        initial_recipe = best_recipe
        for var_recipe in best_recipe.generate_variations():
            if var_recipe.score() > best_score:
                best_recipe = var_recipe
                best_score = var_recipe.score()
        if best_recipe == initial_recipe:
            return best_recipe


def find_optimal_recipe_calorie_constrained(
    ingredients: list[Ingredient], total_qty: int, calorie_count: int
) -> Recipe:
    predicate = lambda r: r.contents()["calories"] == calorie_count

    best_recipe = Recipe.equal_parts(ingredients, total_qty)
    best_score = best_recipe.score() if predicate(best_recipe) else 0

    while True:
        initial_recipe = best_recipe
        for var_recipe in best_recipe.generate_variations_multigen(
            12, lambda r: r.contents()["calories"] == calorie_count
        ):
            if var_recipe.score() > best_score:
                best_recipe = var_recipe
                best_score = var_recipe.score()
        if best_recipe == initial_recipe:
            return best_recipe


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
    @classmethod
    def equal_parts(cls, ingredients: list[Ingredient], total_qty: int) -> Recipe:
        ingredient_count = len(ingredients)
        qty = total_qty // ingredient_count
        additional_qty = total_qty - (ingredient_count * qty)
        return cls(
            (ing, qty + (1 if i < additional_qty else 0)) for i, ing in enumerate(ingredients)
        )

    def __init__(self, ingredients: Iterable[tuple[Ingredient, int]]):
        self.ingredients = tuple(ingredients)

    def __hash__(self) -> int:
        return hash((Recipe, self.ingredients))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Recipe)
            and hash(self) == hash(other)
            and self.ingredients == other.ingredients
        )

    def __repr__(self) -> str:
        contents = ", ".join(f"({i.name}, {qty})" for i, qty in self.ingredients)
        return f"Recipe([{contents}])"

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

    def generate_variations(self) -> Iterator[Recipe]:
        ings, qtys = zip(*self.ingredients)
        for p, m in permutations(range(len(qtys)), 2):
            new_qtys = list(qtys)
            new_qtys[p] += 1
            new_qtys[m] -= 1
            if all(q >= 0 for q in new_qtys):
                yield Recipe(zip(ings, new_qtys))

    def generate_variations_multigen(
        self, target_n: int, predicate: Callable[[Recipe], bool]
    ) -> Iterator[Iterator]:
        """BFS for recipe variations passing the stated predicate."""
        frontier = deque([self])
        visited = set([self])
        result_count = 0
        while frontier and result_count < target_n:
            next_recipe = frontier.pop()
            for variation in next_recipe.generate_variations():
                if variation in visited:
                    continue
                if predicate(variation):
                    yield variation
                    result_count += 1
                visited.add(variation)
                frontier.appendleft(variation)


def generate_quantities(ingredient_count: int, total_qty: int) -> Iterator[tuple[int, ...]]:
    acc = [0] * ingredient_count

    def _inner(idx) -> Iterator[tuple[int, ...]]:
        running_total = sum(acc[:idx])
        if idx == len(acc) - 1:
            acc[idx] = total_qty - running_total
            yield tuple(acc)
            return
        for qty in range(0, total_qty - running_total + 1):
            acc[idx] = qty
            yield from _inner(idx + 1)

    return _inner(0)


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
def recipe(ingredients: list[Ingredient]):
    return Recipe(zip(ingredients, [44, 56]))


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


def test_generate_quantities():
    results = list(generate_quantities(2, 100))
    assert len(results) == 101
    for result in results:
        assert sum(result) == 100


def test_calc_optimal_recipe(ingredients, recipe):
    assert calc_optimal_recipe(ingredients, 100) == recipe
