"""Advent of Code 2023, day 13: https://adventofcode.com/2023/day/13"""
from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from io import StringIO
from typing import IO, Callable, Iterator, Optional, Type

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log

Field = list[str]


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        return self._solve_with(has_reflection_at, input_file)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        return self._solve_parallel(has_reflection_off_by_one, input_file)

    def _solve_with(
        self, reflector: Callable[[Field, int], bool], input_file: Optional[IO] = None
    ) -> int:
        with input_file or self.open_input() as fp:
            return sum(self._score_field(field, reflector) for field in read_fields(fp))

    def _solve_parallel(
        self,
        reflector: Callable[[Field, int], bool],
        input_file: Optional[IO] = None,
        executor_class: Type[Executor] = ThreadPoolExecutor,
    ) -> int:
        pool = executor_class()
        with input_file or self.open_input() as fp:
            scores = pool.map(partial(self._score_field, reflector=reflector), read_fields(fp))
        return sum(scores)

    @staticmethod
    def _score_field(field: Field, reflector: Callable[[Field, int], bool]) -> int:
        result = 0
        for loc in range(1, len(field)):
            if reflector(field, loc):
                result += 100 * loc
                break
        field = flip_field(field)
        for loc in range(1, len(field)):
            if reflector(field, loc):
                result += loc
                break
        return result


def read_fields(file: IO) -> Iterator[Field]:
    next_field = []
    for _row, line in enumerate(file):
        line = line.strip()
        if not line:
            yield next_field
            next_field = []
            continue
        next_field.append(line)
    yield next_field


def flip_field(field: Field) -> Field:
    return ["".join(chars) for chars in zip(*field)]


def has_reflection_at(field: Field, loc: int) -> bool:
    log.debug("Comparing %s == %s", field[:loc][::-1], field[loc:])
    return all(left == right for left, right in zip(field[:loc][::-1], field[loc:]))


def has_reflection_off_by_one(field: Field, loc: int) -> bool:
    log.debug("Comparing %s == %s", field[:loc][::-1], field[loc:])
    return (
        sum(
            1
            for left, right in zip(field[:loc][::-1], field[loc:])
            for lc, rc in zip(left, right)
            if lc != rc
        )
        == 1
    )


SAMPLE_INPUTS = [
    """\
#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#
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
    assert solution.solve_part_one(sample_input) == 405


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 400


def test_read_fields(sample_input):
    results = list(read_fields(sample_input))
    assert results == [
        [
            "#.##..##.",
            "..#.##.#.",
            "##......#",
            "##......#",
            "..#.##.#.",
            "..##..##.",
            "#.#.##.#.",
        ],
        [
            "#...##..#",
            "#....#..#",
            "..##..###",
            "#####.##.",
            "#####.##.",
            "..##..###",
            "#....#..#",
        ],
    ]


def test_flip_field(sample_input):
    result = [flip_field(field) for field in read_fields(sample_input)]
    assert result == [
        [
            "#.##..#",
            "..##...",
            "##..###",
            "#....#.",
            ".#..#.#",
            ".#..#.#",
            "#....#.",
            "##..###",
            "..##...",
        ],
        [
            "##.##.#",
            "...##..",
            "..####.",
            "..####.",
            "#..##..",
            "##....#",
            "..####.",
            "..####.",
            "###..##",
        ],
    ]


@pytest.mark.parametrize(
    ("field_index", "flipped", "loc", "expected"),
    [
        (0, True, 5, True),
        (1, False, 4, True),
    ],
)
def test_has_reflection_at(sample_input, field_index: int, flipped: bool, loc: int, expected: int):
    fields = list(read_fields(sample_input))
    field = fields[field_index]
    if flipped:
        field = flip_field(field)
    assert has_reflection_at(field, loc) is expected
