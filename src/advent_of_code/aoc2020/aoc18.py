"""Advent of Code 2020, day 18: https://adventofcode.com/2020/day/18"""
from abc import abstractmethod
from dataclasses import dataclass
from io import StringIO
from itertools import takewhile
from typing import Generator, Iterator

import parsy as ps
import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(18, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return sum(evaluate(tokenize(line)) for line in f)

    def solve_part_two(self) -> int:
        result = 0
        with self.open_input() as f:
            for line in f:
                tokens = list(tokenize(line))
                result += evaluate_2(tokens).eval()
        return result


def tokenize(text: str) -> Iterator[str]:
    while text:
        match text[0]:
            case "+" | "-" | "*" | "/" | "(" | ")":
                yield text[0]
                text = text[1:]
            case n if n.isnumeric():
                val = "".join(takewhile(lambda v: v.isnumeric(), iter(text)))
                yield val
                text = text[len(val) :]
            case _:
                text = text[1:]


def evaluate(tokens: Iterator[str]) -> int:
    acc = 0
    op = "+"

    def apply(n):
        nonlocal acc
        match op:
            case "+":
                acc += n
            case "*":
                acc *= n
            case _:
                raise ValueError(f"Unknown operation: {op}")

    for token in tokens:
        match token:
            case n if n.isnumeric():
                apply(int(n))
            case "(":
                apply(evaluate(tokens))
            case ")":
                return acc
            case _:
                op = token
    return acc


def lexer(text: str):
    whitespace = ps.regex(r"\s*")
    integer = ps.digit.at_least(1).concat()
    parser = whitespace >> ((integer | ps.regex(r"[+*()]")) << whitespace).many()
    return parser.parse(text)


class Expr:
    @abstractmethod
    def eval(self) -> int:
        pass


@dataclass
class Addition(Expr):
    left: Expr
    right: Expr

    def eval(self) -> int:
        return self.left.eval() + self.right.eval()


@dataclass
class Multiplication(Expr):
    left: Expr
    right: Expr

    def eval(self) -> int:
        return self.left.eval() * self.right.eval()


@dataclass
class Number(Expr):
    value: int

    def eval(self) -> int:
        return self.value


def evaluate_2(tokens: list[str]) -> Expr:
    """Adapted from https://parsy.readthedocs.io/en/latest/howto/lexing.html#calculator"""

    @ps.generate
    def additive() -> Generator[ps.Parser, Expr, Expr]:
        res: Expr = yield simple
        sign: ps.Parser = ps.match_item("+")
        while True:
            operation: str = yield sign | ps.success("")
            if not operation:
                break
            operand: Expr = yield simple
            res = Addition(res, operand)
        return res

    @ps.generate
    def multiplicative() -> Generator[ps.Parser, Expr, Expr]:
        res: Expr = yield additive
        op: ps.Parser = ps.match_item("*")
        while True:
            operation: str = yield op | ps.success("")
            if not operation:
                break
            operand: Expr = yield additive
            res = Multiplication(res, operand)
        return res

    @ps.generate
    def number() -> Generator[ps.Parser, str, Expr]:
        value: str = yield ps.test_item(lambda n: n.isnumeric(), "number")
        return Number(int(value))

    lparen = ps.match_item("(")
    rparen = ps.match_item(")")
    expr = multiplicative
    simple = (lparen >> expr << rparen) | number

    return expr.parse(tokens)


SAMPLE_INPUTS = [
    "1 + 2 * 3 + 4 * 5 + 6",
    "1 + (2 * 3) + (4 * (5 + 6))",
    "2 * 3 + (4 * 5)",
    "5 + (8 * 3 + 9 + 3 * 4 * 3)",
    "5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))",
    "((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2",
    "765 + 42",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 71), (1, 51), (2, 26), (3, 437), (4, 12240), (5, 13632)],
    indirect=["sample_input"],
)
def test_evaluate(sample_input, expected):
    tokens = tokenize(sample_input.readline())
    assert evaluate(tokens) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [
        (0, ["1", "+", "2", "*", "3", "+", "4", "*", "5", "+", "6"]),
        (2, ["2", "*", "3", "+", "(", "4", "*", "5", ")"]),
        (6, ["765", "+", "42"]),
    ],
    indirect=["sample_input"],
)
def test_tokenize(sample_input, expected):
    assert list(tokenize(sample_input.readline())) == expected


@pytest.mark.parametrize(
    ("sample_input", "expected"),
    [(0, 231), (1, 51), (2, 46), (3, 1445), (4, 669060), (5, 23340)],
    indirect=["sample_input"],
)
def test_evaluate_2(sample_input, expected):
    tokens = list(tokenize(sample_input.readline()))
    assert evaluate_2(tokens).eval() == expected


@pytest.mark.parametrize("sample_input", list(range(len(SAMPLE_INPUTS))), indirect=True)
def test_tokenize_equivalent_to_lexer(sample_input):
    line = sample_input.readline()
    assert lexer(line) == list(tokenize(line))
