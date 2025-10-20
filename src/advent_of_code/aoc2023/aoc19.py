"""Advent of Code 2023, day 19: https://adventofcode.com/2023/day/19"""

from __future__ import annotations

import operator
import re
from collections import deque
from dataclasses import dataclass, field, replace
from io import StringIO
from typing import IO, Callable, ClassVar, Iterator, Optional, Sequence

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(19, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            workflows = {w.id: w for w in read_workflows(fp)}
            parts = list(read_parts(fp))

        result = 0
        for part in parts:
            disposition = "in"
            while disposition not in ("A", "R"):
                disposition = workflows[disposition].apply(part)
            if disposition == "A":
                result += part.calc_rating_sum()

        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            workflows = {w.id: w for w in read_workflows(fp)}
        accepted_parts: list[HypotheticalPart] = []
        parts_bin = deque([(HypotheticalPart(), "in")])
        while parts_bin:
            part, disposition = parts_bin.popleft()
            if disposition == "A":
                accepted_parts.append(part)
            elif disposition == "R":
                pass
            else:
                parts_bin.extend(workflows[disposition].apply_hypothetical(part))
        return sum(p.count_possibilities() for p in accepted_parts)


@dataclass(frozen=True)
class Part:
    x: int
    m: int
    a: int
    s: int

    @classmethod
    def from_str(cls, text: str) -> Part:
        text = text.lstrip("{").rstrip("}")
        return eval(f"Part({text})")  # noqa: S307

    def calc_rating_sum(self):
        return self.x + self.m + self.a + self.s


@dataclass(frozen=True)
class HypotheticalPart:
    x: range = field(default=range(1, 4001))
    m: range = field(default=range(1, 4001))
    a: range = field(default=range(1, 4001))
    s: range = field(default=range(1, 4001))

    def count_possibilities(self) -> int:
        return len(self.x) * len(self.m) * len(self.a) * len(self.s)

    def partition(
        self, criterion: Criterion
    ) -> tuple[Optional[HypotheticalPart], Optional[HypotheticalPart]]:
        current_range = getattr(self, criterion.property)
        if criterion.operation == "<" and criterion.limit <= current_range.start:
            # whole range fails
            return None, self
        if criterion.operation == "<" and criterion.limit >= current_range.stop:
            # whole range passes
            return self, None
        if criterion.operation == ">" and criterion.limit >= current_range.stop - 1:
            # whole range fails
            return None, self
        if criterion.operation == ">" and criterion.limit < current_range.start:
            # whole range passes
            return self, None
        if criterion.operation == "<":
            left_range = range(current_range.start, criterion.limit)
            right_range = range(criterion.limit, current_range.stop)
            return replace(self, **{criterion.property: left_range}), replace(
                self, **{criterion.property: right_range}
            )
        left_range = range(current_range.start, criterion.limit + 1)
        right_range = range(criterion.limit + 1, current_range.stop)
        return replace(self, **{criterion.property: right_range}), replace(
            self, **{criterion.property: left_range}
        )


@dataclass(frozen=True)
class Criterion:
    property: str
    operation: str
    limit: int

    OPERATORS: ClassVar[dict[str, Callable]] = {"<": operator.lt, ">": operator.gt}

    @classmethod
    def from_str(cls, text: str):
        prop, op_str, limit_str = re.match(r"([xmas])([<>])(\d+)", text).groups()
        return cls(prop, op_str, int(limit_str))

    def __call__(self, part: Part) -> bool:
        return self.OPERATORS[self.operation](getattr(part, self.property), self.limit)


@dataclass(frozen=True)
class Rule:
    disposition: str
    criterion: Optional[Criterion] = None

    @classmethod
    def from_str(cls, text: str) -> Rule:
        if ":" not in text:
            return cls(disposition=text)
        criterion_str, disposition = text.split(":")
        return cls(disposition, Criterion.from_str(criterion_str))


@dataclass(frozen=True)
class Workflow:
    id: str
    rules: Sequence[Rule]

    @classmethod
    def from_str(cls, text: str) -> Workflow:
        workflow_id, rules_str = re.match(r"([a-z]+)\{(.+)\}", text).groups()
        rules = tuple(Rule.from_str(rule_str) for rule_str in rules_str.split(","))
        return cls(id=workflow_id, rules=rules)

    def apply(self, part: Part) -> str:
        for rule in self.rules:
            if rule.criterion is None or rule.criterion(part):
                return rule.disposition
        raise ValueError("No matching rule found for %s in workflow %s.", part, self)

    def apply_hypothetical(self, part: HypotheticalPart) -> Iterator[tuple[HypotheticalPart, str]]:
        remaining_part = part
        for rule in self.rules:
            if not remaining_part:
                break
            if rule.criterion:
                matched_part, remaining_part = remaining_part.partition(rule.criterion)
                if matched_part:
                    yield matched_part, rule.disposition
            else:
                yield remaining_part, rule.disposition


def read_workflows(file: IO) -> Iterator[Workflow]:
    while line := file.readline().strip():
        yield Workflow.from_str(line)


def read_parts(file: IO) -> Iterator[Part]:
    for line in file:
        yield Part.from_str(line.strip())


SAMPLE_INPUTS = [
    """\
px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 19114


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 167409079868000
