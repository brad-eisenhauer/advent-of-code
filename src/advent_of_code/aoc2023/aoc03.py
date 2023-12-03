"""Advent of Code 2023, day 3: https://adventofcode.com/2023/day/3"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from itertools import chain, product
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(3, 2023, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            schematic = Schematic.read(fp)
        return schematic.calc_sum_of_part_numbers()

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            schematic = Schematic.read(fp)
        return schematic.calc_sum_of_gear_ratios()


Vector = tuple[int, ...]


@dataclass(frozen=True)
class SchematicNumber:
    value: int
    covers: frozenset[Vector]


@dataclass(frozen=True)
class SchematicSymbol:
    symbol: str
    loc: Vector


@dataclass
class Schematic:
    numbers: set[SchematicNumber]
    symbols: set[SchematicSymbol]

    @classmethod
    def read(cls, file: IO) -> Schematic:
        numbers: set[SchematicNumber] = set()
        symbols: set[Vector] = set()

        for row, line in enumerate(file):
            line = line.strip()
            for match in re.finditer(r"\d+", line):
                value = int(match.group(0))
                covers = frozenset((row, col) for col in range(*match.span()))
                numbers.add(SchematicNumber(value, covers))
            for match in re.finditer(r"[^\d\.]", line):
                symbols.add(SchematicSymbol(match.group(), (row, match.start())))

        return cls(numbers, symbols)

    def find_part_numbers(self) -> set[SchematicNumber]:
        number_map = self.build_number_map()
        symbol_adjacent_locs = set(
            chain.from_iterable(
                product((row - 1, row, row + 1), (col - 1, col, col + 1))
                for sym in self.symbols
                for (row, col) in (sym.loc,)
            )
        )
        adjacent_locs = symbol_adjacent_locs & number_map.keys()
        return set(number_map[loc] for loc in adjacent_locs)

    def find_adjacent_parts(
        self, loc: Vector, number_map: Optional[dict[Vector, SchematicNumber]] = None
    ) -> set[SchematicNumber]:
        if number_map is None:
            number_map = self.build_number_map()
        row, col = loc
        symbol_adjacent_locs = product((row - 1, row, row + 1), (col - 1, col, col + 1))
        adjacent_locs = symbol_adjacent_locs & number_map.keys()
        return set(number_map[loc] for loc in adjacent_locs)

    def build_number_map(self) -> dict[Vector, SchematicNumber]:
        return {loc: num for num in self.numbers for loc in num.covers}

    def calc_sum_of_part_numbers(self) -> int:
        return sum(n.value for n in self.find_part_numbers())

    def find_gears(self) -> Iterator[Vector, int, int]:
        number_map = self.build_number_map()
        for sym in self.symbols:
            if sym.symbol != "*":
                continue
            adjacent_parts = self.find_adjacent_parts(sym.loc, number_map)
            if len(adjacent_parts) == 2:
                yield sym.loc, *(p.value for p in adjacent_parts)

    def calc_sum_of_gear_ratios(self) -> int:
        return sum(a * b for _, a, b in self.find_gears())


SAMPLE_INPUTS = [
    """\
467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as fp:
        yield fp


def test_read_schematic(sample_input):
    result = Schematic.read(sample_input)
    expected = Schematic(
        numbers={
            SchematicNumber(467, covers=frozenset([(0, 0), (0, 1), (0, 2)])),
            SchematicNumber(114, covers=frozenset([(0, 5), (0, 6), (0, 7)])),
            SchematicNumber(35, covers=frozenset([(2, 2), (2, 3)])),
            SchematicNumber(633, covers=frozenset([(2, 6), (2, 7), (2, 8)])),
            SchematicNumber(617, covers=frozenset([(4, 0), (4, 1), (4, 2)])),
            SchematicNumber(58, covers=frozenset([(5, 7), (5, 8)])),
            SchematicNumber(592, covers=frozenset([(6, 2), (6, 3), (6, 4)])),
            SchematicNumber(755, covers=frozenset([(7, 6), (7, 7), (7, 8)])),
            SchematicNumber(664, covers=frozenset([(9, 1), (9, 2), (9, 3)])),
            SchematicNumber(598, covers=frozenset([(9, 5), (9, 6), (9, 7)])),
        },
        symbols={
            SchematicSymbol("*", (1, 3)),
            SchematicSymbol("#", (3, 6)),
            SchematicSymbol("*", (4, 3)),
            SchematicSymbol("+", (5, 5)),
            SchematicSymbol("$", (8, 3)),
            SchematicSymbol("*", (8, 5)),
        },
    )
    assert result == expected


def test_calc_sum_of_part_numbers(sample_input):
    assert Schematic.read(sample_input).calc_sum_of_part_numbers() == 4361


def test_calc_sum_of_gear_ratios(sample_input):
    assert Schematic.read(sample_input).calc_sum_of_gear_ratios() == 467835
