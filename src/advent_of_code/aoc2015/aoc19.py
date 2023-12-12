"""Advent of Code 2015, day 19: https://adventofcode.com/2015/day/19"""
from __future__ import annotations

import re
from io import StringIO
from itertools import product
from typing import IO, Collection, Iterable, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util import grammar as G


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(19, 2015, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            rules, initial = read_input(fp)
        return len(set(generate_next_states(initial, rules)))

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            rules, final = read_input(fp)
        grammar = G.ContextFreeGrammar(
            rules=list(G.Rule(r[0], tuple(tokenize(r[1]))) for r in rules),
            start_symbol="e",
        )
        log.debug("Created grammar with %d rules.", len(grammar.rules))
        text = tokenize(final)
        log.debug("Text has %d symbols.", len(text))
        log.debug("Converting grammar to CNF...")
        cnf_grammar = grammar.to_cnf(
            set(text), product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )
        log.debug("CNF grammar has %d rules.", len(cnf_grammar.rules))
        log.debug("Parsing...")
        cyk_result = G.cyk(text, cnf_grammar)
        log.debug("Parsing complete; calculating minimum weight...")
        return G.min_parse_weight(cyk_result, "e")


def read_input(file: IO) -> tuple[Collection[tuple[str, str]], str]:
    rules: list[tuple[str, str]] = []
    while line := file.readline().strip():
        rules.append(tuple(line.split(" => ")))
    initial_string = file.readline().strip()
    return rules, initial_string


def generate_next_states(text: str, rules: Iterable[tuple[str, str]]) -> Iterator[str]:
    for rule in rules:
        for match in re.finditer(rule[0], text):
            yield text[: match.start()] + rule[1] + text[match.end() :]


def tokenize(text: str) -> list[str]:
    pattern = r"(e|[A-Z][a-z]?)"
    return re.findall(pattern, text)


SAMPLE_INPUTS = [
    """\
H => HO
H => OH
O => HH

HOH
""",
    """\
e => H
e => O
H => HO
H => OH
O => HH

HOH
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 4


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 3


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, ([("H", "HO"), ("H", "OH"), ("O", "HH")], "HOH")),
        (1, ([("e", "H"), ("e", "O"), ("H", "HO"), ("H", "OH"), ("O", "HH")], "HOH")),
    ],
    indirect=["sample_input"],
)
def test_read_input(sample_input, expected):
    assert read_input(sample_input) == expected


def test_generate_next_states(sample_input):
    rules, initial_string = read_input(sample_input)
    assert list(generate_next_states(initial_string, rules)) == [
        "HOOH",
        "HOHO",
        "OHOH",
        "HOOH",
        "HHHH",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HOHOHO", ["H", "O", "H", "O", "H", "O"]),
        ("CRnCaSiRnBSiRnFAr", ["C", "Rn", "Ca", "Si", "Rn", "B", "Si", "Rn", "F", "Ar"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected
