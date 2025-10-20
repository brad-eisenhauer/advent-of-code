"""Advent of Code 2022, day 11: https://adventofcode.com/2022/day/11"""

from __future__ import annotations

import re
from ast import literal_eval
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from io import StringIO
from typing import Callable, ClassVar, Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            monkeys = list(Monkey.read_all(f))
        for _ in range(20):
            for m in monkeys:
                m.take_turn()
        return Monkey.calc_monkey_business()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            monkeys = list(Monkey.read_all(f))
        for _ in range(10000):
            for m in monkeys:
                m.take_turn(worry_reduction=WorryReductionMode.Modulus)
        return Monkey.calc_monkey_business()


class WorryReductionMode(Enum):
    Div3 = auto()
    Modulus = auto()


@dataclass
class Monkey:
    id: int
    items: deque[int]
    operation: Callable[[int], int]
    test: int
    destinations: tuple[int, int]
    inspection_count: int = 0

    monkeys: ClassVar[dict[int, Monkey]] = {}

    @classmethod
    def get_monkey(cls, id: int) -> Monkey:
        return cls.monkeys[id]

    @classmethod
    def read(cls, f: TextIO) -> Optional[Monkey]:
        try:
            id = int(re.search(r"(\d+)", f.readline()).groups()[0])
            items = literal_eval("[" + re.search(r"(\d+)(?:, (\d+))*", f.readline()).group() + "]")
            match re.search(r"= (.+)$", f.readline()).groups()[0].split():
                case ["old", "*", n] if n.isnumeric():

                    def operation(old):
                        return old * int(n)

                case ["old", "*", "old"]:

                    def operation(old):
                        return old * old

                case ["old", "+", n]:

                    def operation(old):
                        return old + int(n)

                case other:
                    raise ValueError(f"Unrecognized expression: '{other}'")
            test = int(re.search(r"(\d+)$", f.readline()).groups()[0])
            dests = [
                int(re.search(r"(\d+)$", f.readline()).groups()[0]),
                int(re.search(r"(\d+)$", f.readline()).groups()[0]),
            ]
            f.readline()  # consume blank line
            return Monkey(
                id=id,
                items=deque(items),
                operation=operation,
                test=test,
                destinations=tuple(reversed(dests)),
            )
        except AttributeError:  # regex fails
            return None

    @classmethod
    def read_all(cls, f: TextIO) -> Iterator[Monkey]:
        while (m := cls.read(f)) is not None:
            yield m

    @classmethod
    def calc_monkey_business(cls) -> int:
        highest, next_highest, *_ = sorted(
            (m.inspection_count for m in cls.monkeys.values()), reverse=True
        )
        return highest * next_highest

    @classmethod
    def get_state(cls) -> tuple[tuple[int, ...]]:
        mod = 1
        for m in cls.monkeys.values():
            mod *= m.test
        return tuple(tuple(i % mod for i in m.items) for m in cls.monkeys.values())

    @classmethod
    @cache
    def modulus(cls) -> int:
        result = 1
        for m in cls.monkeys.values():
            result *= m.test
        return result

    def __post_init__(self):
        self.monkeys[self.id] = self

    def inspect_item(self, worry_reduction: WorryReductionMode) -> None:
        item = self.items.popleft()
        item = self.operation(item)
        match worry_reduction:
            case WorryReductionMode.Div3:
                item //= 3
            case WorryReductionMode.Modulus:
                item %= Monkey.modulus()
        self.inspection_count += 1
        self.throw(item)

    def throw(self, item: int) -> None:
        destination = self.destinations[item % self.test == 0]
        self.get_monkey(destination).receive(item)

    def receive(self, item: int):
        self.items.append(item)

    def take_turn(self, worry_reduction: WorryReductionMode = WorryReductionMode.Div3):
        while self.items:
            self.inspect_item(worry_reduction)


SAMPLE_INPUTS = [
    """\
Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

Monkey 2:
  Starting items: 79, 60, 97
  Operation: new = old * old
  Test: divisible by 13
    If true: throw to monkey 1
    If false: throw to monkey 3

Monkey 3:
  Starting items: 74
  Operation: new = old + 3
  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read_monkeys(sample_input):
    monkeys = list(Monkey.read_all(sample_input))
    assert len(monkeys) == 4


@pytest.mark.parametrize(
    ("rounds", "worry_reduction", "expected"),
    [(20, WorryReductionMode.Div3, 10605), (10000, WorryReductionMode.Modulus, 2713310158)],
)
def test_calc_monkey_business(sample_input, rounds, worry_reduction, expected):
    monkeys = list(Monkey.read_all(sample_input))
    for _ in range(rounds):
        for m in monkeys:
            m.take_turn(worry_reduction)
    assert Monkey.calc_monkey_business() == expected
