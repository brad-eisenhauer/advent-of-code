"""Advent of Code 2022, day 21: https://adventofcode.com/2022/day/21"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Collection, TextIO, Union

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import sign

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            monkeys = read(f)
        return round(calc(monkeys, "root"))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            monkeys = read(f)
        return find_match(monkeys)


Monkeys = dict[str, Union[int, str]]


def read(f: TextIO) -> Monkeys:
    result: Monkeys = {}
    for line in f:
        name, content = line.split(": ")
        try:
            result[name] = int(content)
        except ValueError:
            result[name] = content.rstrip()
    return result


def calc(monkeys: Monkeys, name: str) -> Union[int, float]:
    if isinstance(monkeys[name], (int, float)):
        return monkeys[name]
    match monkeys[name].split():
        case [left, "+", right]:
            return calc(monkeys, left) + calc(monkeys, right)
        case [left, "-", right]:
            return calc(monkeys, left) - calc(monkeys, right)
        case [left, "*", right]:
            return calc(monkeys, left) * calc(monkeys, right)
        case [left, "/", right]:
            return calc(monkeys, left) / calc(monkeys, right)


def simplify(monkeys: Monkeys, key: str, protected_refs: Collection[str] = ()):
    if isinstance(monkeys[key], int):
        return
    left, op, right = monkeys[key].split()
    simplify(monkeys, left, protected_refs)
    simplify(monkeys, right, protected_refs)
    if any(k in protected_refs for k in [left, right]):
        return
    left_val = monkeys[left]
    right_val = monkeys[right]
    if any(isinstance(v, str) for v in [left_val, right_val]):
        return
    match op:
        case "+":
            monkeys[key] = left_val + right_val
        case "-":
            monkeys[key] = left_val - right_val
        case "*":
            monkeys[key] = left_val * right_val
        case "/":
            monkeys[key] = left_val // right_val


def find_match(monkeys: Monkeys) -> int:
    eps = 1e-9

    def is_zero(a: Union[int, float]) -> bool:
        return abs(a) < eps

    calc_count = 0

    simplify(monkeys, "root", protected_refs=("humn",))
    left_key, _, right_key = monkeys["root"].split()
    monkeys["root"] = f"{left_key} - {right_key}"

    guess = 1
    log.debug("Starting limit search from %d", guess)
    last_guess = None
    results = {}
    while True:
        monkeys["humn"] = guess
        result = calc(monkeys, "root")
        calc_count += 1
        log.debug("%d: humn=%d => diff=%.2f", calc_count, guess, result)
        if is_zero(result):
            return guess
        results[guess] = result
        if last_guess in results and sign(result) != sign(results[last_guess]):
            break
        last_guess = guess
        guess *= 10

    lo = min(guess, last_guess)
    hi = max(guess, last_guess)
    log.debug("Starting binary search with bounds: %d, %d", lo, hi)
    while hi >= lo:
        guess = lo + (hi - lo) // 2
        monkeys["humn"] = guess
        result = calc(monkeys, "root")
        calc_count += 1
        log.debug("%d: humn=%d => diff=%.2f", calc_count, guess, result)
        if is_zero(result):
            return guess
        results[guess] = result
        if sign(result) == sign(results[lo]):
            lo = guess
        else:
            hi = guess


SAMPLE_INPUTS = [
    """\
root: pppw + sjmn
dbpl: 5
cczh: sllz + lgvd
zczc: 2
ptdq: humn - dvpt
dvpt: 3
lfqf: 4
humn: 5
ljgn: 2
sjmn: drzm * dbpl
sllz: 4
pppw: cczh / lfqf
lgvd: ljgn * ptdq
drzm: hmdt - zczc
hmdt: 32
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read(sample_input):
    expected = {
        "root": "pppw + sjmn",
        "dbpl": 5,
        "cczh": "sllz + lgvd",
        "zczc": 2,
        "ptdq": "humn - dvpt",
        "dvpt": 3,
        "lfqf": 4,
        "humn": 5,
        "ljgn": 2,
        "sjmn": "drzm * dbpl",
        "sllz": 4,
        "pppw": "cczh / lfqf",
        "lgvd": "ljgn * ptdq",
        "drzm": "hmdt - zczc",
        "hmdt": 32,
    }
    assert read(sample_input) == expected


def test_calc(sample_input):
    monkeys = read(sample_input)
    assert calc(monkeys, "root") == 152


def test_simplify(sample_input):
    expected = {
        "root": "pppw + sjmn",
        "dbpl": 5,
        "cczh": "sllz + lgvd",
        "zczc": 2,
        "ptdq": "humn - dvpt",
        "dvpt": 3,
        "lfqf": 4,
        "humn": 5,
        "ljgn": 2,
        "sjmn": 150,
        "sllz": 4,
        "pppw": "cczh / lfqf",
        "lgvd": "ljgn * ptdq",
        "drzm": 30,
        "hmdt": 32,
    }
    monkeys = read(sample_input)
    simplify(monkeys, "root", ("humn",))
    assert monkeys == expected


def test_find_match(sample_input):
    monkeys = read(sample_input)
    assert find_match(monkeys) == 301
