"""Advent of Code 2023, day 12: https://adventofcode.com/2023/day/12"""
from __future__ import annotations

import re
from functools import cache
from io import StringIO
from typing import IO, Iterator, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as fp:
            for line in fp:
                text, groups = read_input(line)
                result += count_arrangements(pad(text), groups)
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as fp:
            for line in fp:
                text, groups = read_input(line)
                text = "?".join([text] * 5)
                groups = groups * 5
                result += count_arrangements(pad(text), groups)
        return result


def read_input(line: str) -> tuple[str, tuple[int, ...]]:
    pattern, groups_str = line.split()
    groups = tuple(int(n) for n in groups_str.split(","))
    return pattern, groups


def pad(text: str) -> str:
    return text + "."


def unpad(text: str) -> str:
    return text[:-1]


def generate_arrangements(text: str, groups: tuple[int, ...]) -> Iterator[str]:
    if not groups:
        if "#" in text:
            return
        else:
            result = text.replace("?", ".")
            log.debug("Yielding '%s' from '%s' %s", result, text, groups)
            yield result
            return

    pattern = rf"^([\.\?]*?)[\?#]{{{groups[0]}}}[\.\?]"
    match = re.match(pattern, text)
    if match is None:
        return
    result_prefix = match.group(1).replace("?", ".") + "#" * groups[0] + "."
    for sub_arrangement in generate_arrangements(text[match.end() :], groups[1:]):
        result = result_prefix + sub_arrangement
        log.debug("Yielding '%s' from '%s' %s", result, text, groups)
        yield result
    if text[len(match.group(1))] == "#":
        return
    for sub_arrangement in generate_arrangements(text[len(match.group(1)) + 1 :], groups):
        result = text[: len(match.group(1)) + 1].replace("?", ".") + sub_arrangement
        log.debug("Yielding '%s' from '%s' %s", result, text, groups)
        yield result


@cache
def count_arrangements(text: str, groups: tuple[int, ...]) -> int:
    if not groups:
        return 0 if "#" in text else 1

    pattern = rf"^([\.\?]*?)[\?#]{{{groups[0]}}}[\.\?]"
    match = re.match(pattern, text)
    if match is None:
        return 0
    result = count_arrangements(text[match.end() :], groups[1:])
    if text[len(match.group(1))] == "#":
        return result
    result += count_arrangements(text[len(match.group(1)) + 1 :], groups)
    return result


SAMPLE_INPUTS = [
    """\
???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 21


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 525152


def test_count_arrangements(sample_input):
    expected = [1, 4, 1, 1, 4, 10]
    result = []
    for line in sample_input:
        text, groups = read_input(line)
        result.append(count_arrangements(pad(text), groups))
    assert result == expected


@pytest.mark.parametrize(
    ("text", "groups", "expected"),
    [
        ("???.###", (1, 1, 3), {"#.#.###"}),
        (
            ".??..??...?##.",
            (1, 1, 3),
            {".#...#....###.", ".#....#...###.", "..#..#....###.", "..#...#...###."},
        ),
        (
            "#????????.#?#??????",
            (2, 1, 1, 5, 1),
            {
                "##.#.#....#####.#..",
                "##.#.#....#####..#.",
                "##.#.#....#####...#",
                "##.#..#...#####.#..",
                "##.#..#...#####..#.",
                "##.#..#...#####...#",
                "##.#...#..#####.#..",
                "##.#...#..#####..#.",
                "##.#...#..#####...#",
                "##.#....#.#####.#..",
                "##.#....#.#####..#.",
                "##.#....#.#####...#",
                "##..#.#...#####.#..",
                "##..#.#...#####..#.",
                "##..#.#...#####...#",
                "##..#..#..#####.#..",
                "##..#..#..#####..#.",
                "##..#..#..#####...#",
                "##..#...#.#####.#..",
                "##..#...#.#####..#.",
                "##..#...#.#####...#",
                "##...#.#..#####.#..",
                "##...#.#..#####..#.",
                "##...#.#..#####...#",
                "##...#..#.#####.#..",
                "##...#..#.#####..#.",
                "##...#..#.#####...#",
                "##....#.#.#####.#..",
                "##....#.#.#####..#.",
                "##....#.#.#####...#",
                "##.#......#.#####.#",
                "##..#.....#.#####.#",
                "##...#....#.#####.#",
                "##....#...#.#####.#",
                "##.....#..#.#####.#",
                "##......#.#.#####.#",
            },
        ),
    ],
)
def test_generate_arrangements(text, groups, expected):
    assert {unpad(result) for result in generate_arrangements(pad(text), groups)} == expected
