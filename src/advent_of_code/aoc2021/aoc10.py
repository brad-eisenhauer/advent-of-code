""" Advent of Code 2021, Day 10: https://adventofcode.com/2021/day/10 """
from io import StringIO
from typing import Iterator, TextIO

import numpy as np
import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            score, _ = calc_scores(f)
        return score

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            _, score = calc_scores(f)
        return score


class IllegalCharError(SyntaxError):
    def __init__(self, position: int, illegal_char: str):
        super().__init__(f"Illegal character '{illegal_char}' found at position {position}.")
        self.position = position
        self.illegal_char = illegal_char


class IncompleteLineError(SyntaxError):
    def __init__(self, stack: list[str]):
        super().__init__(f"Line incomplete; remaining stack: '{''.join(stack)}'.")
        self.stack = stack


OPENERS_AND_CLOSERS = {"(": ")", "[": "]", "{": "}", "<": ">"}
ILLEGAL_CLOSER_SCORES = {")": 3, "]": 57, "}": 1197, ">": 25137}
COMPLETION_SCORES = {"(": 1, "[": 2, "{": 3, "<": 4}


def calc_scores(fp: TextIO) -> tuple[int, int]:
    corrupted_score = 0
    incomplete_scores = []
    for line in fp:
        try:
            validate_line(line.strip())
        except IncompleteLineError as e:
            line_score = 0
            for char in reversed(e.stack):
                line_score = 5 * line_score + COMPLETION_SCORES[char]
            incomplete_scores.append(line_score)
        except IllegalCharError as e:
            corrupted_score += ILLEGAL_CLOSER_SCORES[e.illegal_char]

    return corrupted_score, int(np.median(incomplete_scores))


def validate_line(line: str):
    stack = []
    for i, char in enumerate(line):
        if char in OPENERS_AND_CLOSERS:
            stack.append(char)
        else:
            try:
                expected = OPENERS_AND_CLOSERS[stack.pop()]
                if char != expected:
                    raise IllegalCharError(i, char)
            except IndexError:
                return IllegalCharError(i, char)

    if len(stack) > 0:
        raise IncompleteLineError(stack)


TEST_INPUT = """[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]
"""


@pytest.fixture()
def sample_input() -> Iterator[TextIO]:
    with StringIO(TEST_INPUT) as fp:
        yield fp


def test_calc_score():
    with StringIO(TEST_INPUT) as fp:
        score = calc_scores(fp)

    assert score == (26397, 288957)
