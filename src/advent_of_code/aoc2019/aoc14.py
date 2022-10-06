"""Advent of Code 2019, day 14: https://adventofcode.com/2019/day/14"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(14, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            reactor = Reactor.from_file(f)
        requirements, _ = reactor.requirements("FUEL", 1)
        return requirements["ORE"]


@dataclass
class Recipe:
    reagents: dict[str, int]
    product: tuple[str, int]

    @classmethod
    def from_string(cls, text: str) -> Recipe:
        reagents_str, product_str = re.match(r"(.*) => (.*)", text).groups()
        reagents = {
            sym: int(qty)
            for reagent in reagents_str.split(",")
            for qty, sym in (reagent.strip().split(),)
        }
        qty, product = product_str.split()
        return Recipe(reagents, (product, int(qty)))


@dataclass
class Reactor:
    recipes: dict[str, Recipe]

    @classmethod
    def from_file(cls, f: TextIO) -> Reactor:
        recipes = {}
        for line in f.readlines():
            recipe = Recipe.from_string(line.strip())
            recipes[recipe.product[0]] = recipe
        return Reactor(recipes)

    def requirements(self, product: str, qty: int) -> tuple[dict[str, int], dict[str, int]]:
        excess = defaultdict(int)
        intermediates: deque[tuple[str, int]] = deque()
        intermediates.append((product, qty))
        result = defaultdict(int)
        while intermediates:
            p, q = intermediates.pop()
            if p in excess:
                q_adjustment = min(q, excess[p])
                excess[p] -= q_adjustment
                q -= q_adjustment
            if p in self.recipes:
                rec = self.recipes[p]
                _, rec_q = rec.product
                multiple = int(math.ceil(q / rec_q))
                prod_q = rec_q * multiple
                if prod_q > q:
                    excess[p] += prod_q - q
                for rea, rea_q in rec.reagents.items():
                    intermediates.append((rea, multiple * rea_q))
            else:
                result[p] += q

        return dict(result), dict(excess)


SAMPLE_INPUTS = [
    """\
10 ORE => 10 A
1 ORE => 1 B
7 A, 1 B => 1 C
7 A, 1 C => 1 D
7 A, 1 D => 1 E
7 A, 1 E => 1 FUEL
""",
    """\
9 ORE => 2 A
8 ORE => 3 B
7 ORE => 5 C
3 A, 4 B => 1 AB
5 B, 7 C => 1 BC
4 C, 1 A => 1 CA
2 AB, 3 BC, 4 CA => 1 FUEL
""",
    """\
157 ORE => 5 NZVS
165 ORE => 6 DCFZ
44 XJWVT, 5 KHKGT, 1 QDVJ, 29 NZVS, 9 GPVTF, 48 HKGWZ => 1 FUEL
12 HKGWZ, 1 GPVTF, 8 PSHF => 9 QDVJ
179 ORE => 7 PSHF
177 ORE => 5 HKGWZ
7 DCFZ, 7 PSHF => 2 XJWVT
165 ORE => 2 GPVTF
3 DCFZ, 7 NZVS, 5 HKGWZ, 10 PSHF => 8 KHKGT
""",
    """\
2 VPVL, 7 FWMGM, 2 CXFTF, 11 MNCFX => 1 STKFG
17 NVRVD, 3 JNWZP => 8 VPVL
53 STKFG, 6 MNCFX, 46 VJHF, 81 HVMC, 68 CXFTF, 25 GNMV => 1 FUEL
22 VJHF, 37 MNCFX => 5 FWMGM
139 ORE => 4 NVRVD
144 ORE => 7 JNWZP
5 MNCFX, 7 RFSQX, 2 FWMGM, 2 VPVL, 19 CXFTF => 3 HVMC
5 VJHF, 7 MNCFX, 9 VPVL, 37 CXFTF => 6 GNMV
145 ORE => 6 MNCFX
1 NVRVD => 8 CXFTF
1 VJHF, 6 MNCFX => 4 RFSQX
176 ORE => 6 VJHF
""",
    """\
171 ORE => 8 CNZTR
7 ZLQW, 3 BMBT, 9 XCVML, 26 XMNCP, 1 WPTQ, 2 MZWV, 1 RJRHP => 4 PLWSL
114 ORE => 4 BHXH
14 VRPVC => 6 BMBT
6 BHXH, 18 KTJDG, 12 WPTQ, 7 PLWSL, 31 FHTLT, 37 ZDVW => 1 FUEL
6 WPTQ, 2 BMBT, 8 ZLQW, 18 KTJDG, 1 XMNCP, 6 MZWV, 1 RJRHP => 6 FHTLT
15 XDBXC, 2 LTCX, 1 VRPVC => 6 ZLQW
13 WPTQ, 10 LTCX, 3 RJRHP, 14 XMNCP, 2 MZWV, 1 ZLQW => 1 ZDVW
5 BMBT => 4 WPTQ
189 ORE => 9 KTJDG
1 MZWV, 17 XDBXC, 3 XCVML => 2 XMNCP
12 VRPVC, 27 CNZTR => 2 XDBXC
15 KTJDG, 12 BHXH => 5 XCVML
3 BHXH, 2 VRPVC => 7 MZWV
121 ORE => 7 VRPVC
7 XCVML => 6 RJRHP
5 BHXH, 4 VRPVC => 5 LTCX
""",
]


def test_parse_reactor():
    expected = Reactor(
        recipes={
            "A": Recipe({"ORE": 10}, ("A", 10)),
            "B": Recipe({"ORE": 1}, ("B", 1)),
            "C": Recipe({"A": 7, "B": 1}, ("C", 1)),
            "D": Recipe({"A": 7, "C": 1}, ("D", 1)),
            "E": Recipe({"A": 7, "D": 1}, ("E", 1)),
            "FUEL": Recipe({"A": 7, "E": 1}, ("FUEL", 1)),
        }
    )
    with StringIO(SAMPLE_INPUTS[0]) as f:
        result = Reactor.from_file(f)
    assert result == expected


@pytest.mark.parametrize(
    ("product", "qty", "expected"),
    [
        ("B", 1, ({"ORE": 1}, {})),
        ("C", 1, ({"ORE": 11}, {"A": 3})),
        ("D", 1, ({"ORE": 21}, {"A": 6})),
        ("E", 1, ({"ORE": 31}, {"A": 9})),
        ("FUEL", 1, ({"ORE": 31}, {"A": 2})),
    ]
)
def test_reactor_requirements(product, qty, expected):
    with StringIO(SAMPLE_INPUTS[0]) as f:
        reactor = Reactor.from_file(f)
    result = reactor.requirements(product, qty)
    assert result == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (SAMPLE_INPUTS[0], 31),
        (SAMPLE_INPUTS[1], 165),
        (SAMPLE_INPUTS[2], 13312),
    ],
)
def test_reactor(sample_input, expected):
    with StringIO(sample_input) as f:
        reactor = Reactor.from_file(f)
    requirements, _ = reactor.requirements("FUEL", 1)
    assert requirements["ORE"] == expected
