"""Advent of Code 2015, day 7: https://adventofcode.com/2015/day/7"""
from __future__ import annotations

from copy import deepcopy
from io import StringIO
from typing import TextIO, Union

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            circuit = parse_circuit(f)
        return value_of(circuit, "a")

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            circuit_a = parse_circuit(f)
        circuit_b = deepcopy(circuit_a)
        circuit_b["b"] = value_of(circuit_a, "a")
        return value_of(circuit_b, "a")


Circuit = dict[str, Union[str, int]]
LIMIT = 2**16 - 1


def parse_circuit(f: TextIO) -> Circuit:
    result = {}
    for line in f:
        expr, wire = line.split(" -> ")
        wire = wire.strip()
        if expr.isnumeric():
            result[wire] = int(expr)
        else:
            result[wire] = expr
    return result


def value_of(circuit: Circuit, wire: str) -> int:
    if wire.isnumeric():
        return int(wire)

    if isinstance(circuit[wire], int):
        return circuit[wire]

    expr = circuit[wire]
    match expr.split():
        case [left, "AND", right]:
            result = value_of(circuit, left) & value_of(circuit, right)
        case [left, "OR", right]:
            result = value_of(circuit, left) | value_of(circuit, right)
        case [left, "RSHIFT", n]:
            result = value_of(circuit, left) >> int(n)
        case [left, "LSHIFT", n]:
            result = (value_of(circuit, left) << int(n)) & LIMIT
        case ["NOT", wire]:
            result = (~value_of(circuit, wire)) & LIMIT
        case [val] if val.isnumeric():
            result = int(val)
        case [wire]:
            result = value_of(circuit, wire)
        case other:
            raise ValueError(f"Unrecognized operation: '{other}'")

    circuit[wire] = result
    return result


SAMPLE_INPUTS = [
    """\
123 -> x
456 -> y
x AND y -> d
x OR y -> e
x LSHIFT 2 -> f
y RSHIFT 2 -> g
NOT x -> h
NOT y -> i
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture()
def sample_circuit(sample_input) -> Circuit:
    return parse_circuit(sample_input)


def test_parse_circuit(sample_input):
    circuit = parse_circuit(sample_input)
    assert circuit == {
        "x": 123,
        "y": 456,
        "d": "x AND y",
        "e": "x OR y",
        "f": "x LSHIFT 2",
        "g": "y RSHIFT 2",
        "h": "NOT x",
        "i": "NOT y",
    }


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("x", 123),
        ("y", 456),
        ("d", 72),
        ("e", 507),
        ("f", 492),
        ("g", 114),
        ("h", 65412),
        ("i", 65079),
    ],
)
def test_value_of(sample_circuit, wire, expected):
    assert value_of(sample_circuit, wire) == expected
