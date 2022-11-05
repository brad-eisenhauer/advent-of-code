"""Advent of Code 2020, day 16: https://adventofcode.com/2020/day/16"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from itertools import product
from typing import Iterator, Optional, TextIO

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            rules = list(read_rules(f))
            [_] = read_tickets(f)
            tickets = read_tickets(f)
            return sum(v for t in tickets for v in validate_ticket(t, rules))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            rules = list(read_rules(f))
            [my_ticket] = read_tickets(f)
            nearby_tickets = list(read_tickets(f))
        valid_tickets = [my_ticket, *(t for t in nearby_tickets if not validate_ticket(t, rules))]
        fields = decode_fields_take_2(valid_tickets, rules)
        log.debug("Solution: %s", fields)
        result = 1
        for field, value in zip(fields, my_ticket):
            if "departure" in field:
                result *= value
        return result


@dataclass(frozen=True)
class Rule:
    name: str
    ranges: list[range] = field(hash=False)

    @classmethod
    def parse(cls, line: str) -> Rule:
        name, ranges = line.split(":")
        ranges = [
            range(int(lo), int(hi) + 1)
            for rng in ranges.split(" or ")
            for lo, hi in (rng.split("-"),)
        ]
        return cls(name, ranges)

    def validate(self, value: int) -> bool:
        return any(value in rng for rng in self.ranges)


def read_rules(f: TextIO) -> Iterator[Rule]:
    for line in f:
        line = line.strip()
        if not line:
            return
        yield Rule.parse(line)


def read_tickets(f: TextIO) -> Iterator[list[int]]:
    # throw away header line
    _ = f.readline()
    for line in f:
        line = line.strip()
        if not line:
            return
        yield [int(n) for n in line.split(",")]


def validate_ticket(ticket: list[int], rules: list[Rule]) -> list[int]:
    invalid_vals = []
    for value in ticket:
        if not any(r.validate(value) for r in rules):
            invalid_vals.append(value)
    return invalid_vals


def decode_fields(
    valid_tickets: list[list[int]],
    rules: list[Rule],
    field_index: int = 0,
    valid_rules: dict[int, set[Rule]] = None,
    call_stats: list = None,
) -> Optional[list[str]]:
    field_count = len(valid_tickets[0])
    if valid_rules is None:
        valid_rules = defaultdict(set)
        for rule, index in product(rules, range(field_count)):
            if all(rule.validate(ticket[index]) for ticket in valid_tickets):
                valid_rules[index].add(rule)
        log.debug(
            "Decoding %d fields (%d rules) on %d tickets.",
            field_count,
            len(rules),
            len(valid_tickets),
        )

    if call_stats is None:
        call_stats = [time.monotonic(), 0]
    elif call_stats[1] % 1_000_000 == 0:
        log.debug("Call rate: %.1f /ms", call_stats[1] / (time.monotonic() - call_stats[0]) / 1000)
    call_stats[1] += 1

    if field_index >= field_count:
        return []
    for rule in rules:
        if rule in valid_rules[field_index]:
            remaining_indexes = range(field_index + 1, field_count)
            next_rules = sorted(
                (r for r in rules if r is not rule),
                key=lambda r: sum(1 for i in remaining_indexes if r in valid_rules[i]),
            )
            decoded_tail = decode_fields(
                valid_tickets,
                next_rules,
                field_index + 1,
                valid_rules,
                call_stats,
            )
            if decoded_tail is not None:
                return [rule.name, *decoded_tail]
    return None


def decode_fields_take_2(valid_tickets: list[list[int]], rules: list[Rule]):
    field_count = len(valid_tickets[0])
    assert len(rules) == field_count

    results = {
        (rule, index): all(rule.validate(ticket[index]) for ticket in valid_tickets)
        for rule, index in product(rules, range(field_count))
    }
    valid_indexes: dict[Rule, set[int]] = {
        rule: {i for i in range(field_count) if results[(rule, i)]} for rule in rules
    }
    valid_rules: dict[int, set[Rule]] = {
        index: {r for r in rules if results[(r, index)]} for index in range(field_count)
    }

    while True:
        progress_flag = False
        for rule, indexes in valid_indexes.items():
            try:
                (index,) = indexes
            except ValueError:
                pass
            else:
                # invalidate all other rules for this index, and this index for all other rules
                for r in rules:
                    if r is not rule and r in valid_rules[index]:
                        progress_flag = True
                        valid_rules[index].remove(r)
                        valid_indexes[r].remove(index)

        for index, rules in valid_rules.items():
            try:
                (rule,) = rules
            except ValueError:
                pass
            else:
                # invalidate all other indexes for this rule, and this rule for all other indexes
                for i in range(field_count):
                    if i != index and i in valid_indexes[rule]:
                        progress_flag = True
                        valid_indexes[rule].remove(i)
                        valid_rules[i].remove(rule)

        if all(len(valid_rules[i]) == 1 for i in range(field_count)):
            result = [r.name for i in range(field_count) for r in valid_rules[i]]
            return result
        if not progress_flag:
            raise ValueError("Stuck in infinite loop.")


SAMPLE_INPUTS = [
    """\
class: 1-3 or 5-7
row: 6-11 or 33-44
seat: 13-40 or 45-50

your ticket:
7,1,14

nearby tickets:
7,3,47
40,4,50
55,2,20
38,6,12
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_read_rules(sample_input):
    rules = list(read_rules(sample_input))
    expected = [
        Rule("class", [range(1, 4), range(5, 8)]),
        Rule("row", [range(6, 12), range(33, 45)]),
        Rule("seat", [range(13, 41), range(45, 51)]),
    ]
    assert rules == expected


def test_read_tickets(sample_input):
    _ = list(read_rules(sample_input))
    [my_ticket] = read_tickets(sample_input)
    assert my_ticket == [7, 1, 14]
    nearby_tickets = list(read_tickets(sample_input))
    assert len(nearby_tickets) == 4


def test_validate_ticket(sample_input):
    rules = list(read_rules(sample_input))
    _ = list(read_tickets(sample_input))
    nearby_tickets = list(read_tickets(sample_input))
    invalid_vals = [v for ticket in nearby_tickets for v in validate_ticket(ticket, rules)]
    assert invalid_vals == [4, 55, 12]


def test_decode_fields(sample_input):
    rules = list(read_rules(sample_input))
    [my_ticket] = read_tickets(sample_input)
    nearby_tickets = list(read_tickets(sample_input))
    valid_tickets = [my_ticket, *(t for t in nearby_tickets if not validate_ticket(t, rules))]
    result = decode_fields_take_2(valid_tickets, rules)
    assert result == ["row", "class", "seat"]
