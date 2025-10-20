"""Advent of Code 2016, day 21: https://adventofcode.com/2016/day/21"""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from dataclasses import dataclass
from io import StringIO
from itertools import permutations
from typing import IO, Callable, Optional, Self, ClassVar

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None, initial: str = "abcdefgh") -> str:
        with input_file or self.open_input() as fp:
            instructions = fp.readlines()
        scrambler = Scrambler.from_instructions(instructions)
        return scrambler.scramble(initial)

    def solve_part_two(
        self, input_file: Optional[IO] = None, target: str = "fbgdceah", brute_force: bool = False
    ) -> str:
        with input_file or self.open_input() as fp:
            instructions = fp.readlines()
        scrambler = Scrambler.from_instructions(instructions)
        if brute_force:
            for candidate in permutations("abcdefgh", 8):
                if scrambler.scramble(candidate) == "target":
                    return "".join(candidate)
        return scrambler.unscramble(target)


@dataclass
class Mutator:
    instruction: str

    PATTERN: ClassVar[str]

    @classmethod
    def from_str(cls, text: str) -> Self | None:
        if (match := re.match(cls.PATTERN, text)) is None:
            return None
        return cls(instruction=text.rstrip(), **match.groupdict())

    @abstractmethod
    def __call__(self, text: list[str]) -> None:
        pass

    def reverse(self, text: list[str]) -> None:
        self(text)


def log_mutation(mut_fn: Callable[[list[str]], None]) -> Callable[[list[str]], None]:
    def wrapper(self: Mutator, text: list[str]) -> None:
        log.debug("Mutation: %s", self.instruction)
        log.debug("Before: %s", text)
        mut_fn(self, text)
        log.debug("After: %s", text)

    return wrapper


@dataclass
class PositionSwapper(Mutator):
    index_a: int
    index_b: int

    PATTERN: ClassVar[str] = r"swap position (?P<index_a>\d+) with position (?P<index_b>\d+)"

    def __post_init__(self) -> None:
        self.index_a = int(self.index_a)
        self.index_b = int(self.index_b)

    @log_mutation
    def __call__(self, text: list[str]) -> None:
        text[self.index_a], text[self.index_b] = text[self.index_b], text[self.index_a]


@dataclass
class ValueSwapper(Mutator):
    value_a: str
    value_b: str

    PATTERN: ClassVar[str] = r"swap letter (?P<value_a>[a-z]) with letter (?P<value_b>[a-z])"

    @log_mutation
    def __call__(self, text: list[str]) -> None:
        index_a = text.index(self.value_a)
        index_b = text.index(self.value_b)
        text[index_a] = self.value_b
        text[index_b] = self.value_a


@dataclass
class Rotator(Mutator):
    direction: str
    magnitude: int

    PATTERN: ClassVar[str] = r"rotate (?P<direction>left|right) (?P<magnitude>\d+) steps?"

    def __post_init__(self) -> None:
        self.magnitude = int(self.magnitude)

    @log_mutation
    def __call__(self, text: list[str]) -> None:
        _rotate(text, self.magnitude if self.direction == "left" else -self.magnitude)

    @log_mutation
    def reverse(self, text: list[str]) -> None:
        _rotate(text, self.magnitude if self.direction == "right" else -self.magnitude)


@dataclass
class CharRotator(Mutator):
    char: str

    PATTERN: ClassVar[str] = r"rotate based on position of letter (?P<char>[a-z])"

    @log_mutation
    def __call__(self, text: list[str]) -> str:
        index = text.index(self.char)
        magnitude = 1 + index
        if index >= 4:
            magnitude += 1
        _rotate(text, -magnitude)

    @log_mutation
    def reverse(self, text: list[str]) -> str:
        magnitudes = [-7, 1, -2, 2, -1, 3, 0, 4]  # based on string length == 8
        index = text.index(self.char)
        if magnitudes[index]:
            _rotate(text, magnitudes[index])


@dataclass
class SubstringReverser(Mutator):
    index_a: int
    index_b: int

    PATTERN: ClassVar[str] = r"reverse positions (?P<index_a>\d+) through (?P<index_b>\d+)"

    def __post_init__(self) -> None:
        self.index_a = int(self.index_a)
        self.index_b = int(self.index_b)

    @log_mutation
    def __call__(self, text: list[str]) -> None:
        left_index, right_index = self.index_a, self.index_b
        if left_index > right_index:
            left_index, right_index = right_index, left_index
        start_bound = _to_slice_bound(right_index, len(text))
        end_bound = _to_slice_bound(left_index - 1, len(text))
        text[left_index : right_index + 1] = text[start_bound:end_bound:-1]


@dataclass
class CharMover(Mutator):
    src_index: int
    dest_index: int

    PATTERN: ClassVar[str] = r"move position (?P<src_index>\d+) to position (?P<dest_index>\d+)"

    def __post_init__(self) -> None:
        self.src_index = int(self.src_index)
        self.dest_index = int(self.dest_index)

    @log_mutation
    def __call__(self, text: list[str]) -> None:
        self._impl(text, self.src_index, self.dest_index)

    @log_mutation
    def reverse(self, text: list[str]) -> None:
        self._impl(text, self.dest_index, self.src_index)

    @staticmethod
    def _impl(text: list[str], src_index: int, dest_index: int) -> None:
        value = text[src_index]
        if src_index < dest_index:
            text[src_index:dest_index] = text[src_index + 1 : dest_index + 1]
        else:
            text[dest_index + 1 : src_index + 1] = text[dest_index:src_index]
        text[dest_index] = value


def _rotate(text: list[str], magnitude: int) -> None:
    magnitude %= len(text)
    if magnitude:
        text[:-magnitude], text[-magnitude:] = text[magnitude:], text[:magnitude]


def _to_slice_bound(index: int, length: int) -> int | None:
    if 0 <= index < length:
        return index
    return None


@dataclass
class Scrambler:
    mutators: list[Mutator]

    MUTATOR_CLASSES: ClassVar[list[type[Mutator]]] = [
        PositionSwapper,
        ValueSwapper,
        Rotator,
        CharRotator,
        SubstringReverser,
        CharMover,
    ]

    @classmethod
    def from_instructions(cls, instructions: list[str]) -> Self:
        mutators: list[Mutator] = []
        for line in instructions:
            for mutator_cls in cls.MUTATOR_CLASSES:
                if (mutator := mutator_cls.from_str(line)) is not None:
                    mutators.append(mutator)
                    break
        return cls(mutators=mutators)

    def scramble(self, text: str | list[str]) -> str:
        if not isinstance(text, list):
            text = list(text)
        for mutator in self.mutators:
            mutator(text)
        return "".join(text)

    def unscramble(self, text: str | list[str]) -> str:
        if not isinstance(text, list):
            text = list(text)
        for mutator in self.mutators[::-1]:
            mutator.reverse(text)
        return "".join(text)


SAMPLE_INPUTS = [
    """\
swap position 4 with position 0
swap letter d with letter b
reverse positions 0 through 4
rotate left 1 step
move position 1 to position 4
move position 3 to position 0
rotate based on position of letter b
rotate based on position of letter d
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
    assert solution.solve_part_one(sample_input, initial="abcde") == "decab"


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == "efghdabc"
