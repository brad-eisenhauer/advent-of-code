"""Advent of Code 2020, day 21: https://adventofcode.com/2020/day/21"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from typing import Iterable

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            labels = (Label.read(line) for line in f)
            return count_non_allergen_ingredients(labels)

    def solve_part_two(self) -> str:
        with self.open_input() as f:
            allergens = identify_allergens(Label.read(line) for line in f)
        return ",".join(i for a, i in sorted(allergens.items()))


@dataclass
class Label:
    ingredients: list[str]
    known_allergens: list[str]

    @classmethod
    def read(cls, text: str) -> Label:
        pattern = re.compile(r"([\w\s]+) \(contains ([\w,\s]+)\)")
        ingredients, known_allergens = pattern.match(text).groups()
        return cls(ingredients.split(), known_allergens.split(", "))


def identify_allergens(labels: Iterable[Label]) -> dict[str, str]:
    possible_ingredients: dict[str, set[str]] = {}
    for label in labels:
        for allergen in label.known_allergens:
            if allergen in possible_ingredients:
                possible_ingredients[allergen] &= set(label.ingredients)
            else:
                possible_ingredients[allergen] = set(label.ingredients)
            if len(possible_ingredients[allergen]) == 1:
                (ingredient,) = possible_ingredients[allergen]
                for a, i in possible_ingredients.items():
                    if a != allergen:
                        i -= {ingredient}
    result: dict[str, str] = {}
    for allergen, ingredients in possible_ingredients.items():
        (ingredient,) = ingredients
        result[allergen] = ingredient
    return result


def find_allergen_free_ingredients(labels: Iterable[Label]) -> set[str]:
    labels = list(labels)
    all_ingredients: set[str] = reduce(
        operator.or_, (set(label.ingredients) for label in labels), set()
    )
    allergen_ingredients = set(identify_allergens(labels).values())
    return all_ingredients - allergen_ingredients


def count_ingredient_appearances(labels: Iterable[Label], ingredients: set[str]) -> int:
    result = 0
    for label in labels:
        result += sum(1 for ingredient in label.ingredients if ingredient in ingredients)
    return result


def count_non_allergen_ingredients(labels: Iterable[Label]) -> int:
    labels = list(labels)
    non_allergen_ingredients = find_allergen_free_ingredients(labels)
    return count_ingredient_appearances(labels, non_allergen_ingredients)


SAMPLE_INPUTS = [
    """\
mxmxvkd kfcds sqjhc nhms (contains dairy, fish)
trh fvjkl sbzzf mxmxvkd (contains dairy)
sqjhc fvjkl (contains soy)
sqjhc mxmxvkd sbzzf (contains fish)
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.mark.parametrize(
    ("line_index", "expected"),
    [(0, (4, 2)), (1, (4, 1)), (2, (2, 1)), (3, (3, 1))],
)
def test_read_label(sample_input, line_index, expected):
    lines = sample_input.readlines()
    label = Label.read(lines[line_index])
    assert (len(label.ingredients), len(label.known_allergens)) == expected


def test_identify_allergens(sample_input):
    result = identify_allergens(Label.read(line) for line in sample_input)
    assert result == {"dairy": "mxmxvkd", "fish": "sqjhc", "soy": "fvjkl"}


def test_count_non_allergen_ingredients(sample_input):
    labels = (Label.read(line) for line in sample_input)
    assert count_non_allergen_ingredients(labels) == 5
