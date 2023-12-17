"""Advent of Code 2016, day 16: https://adventofcode.com/2016/day/16"""
from __future__ import annotations

from functools import cache
from io import StringIO
from itertools import count, islice
from typing import IO, Iterator, Optional, Union

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util import create_groups


class AocSolution(Solution[str, str]):
    def __init__(self, **kwargs):
        super().__init__(16, 2016, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> str:
        with input_file or self.open_input() as fp:
            initial_text = fp.read().strip()
        expanded_text = fill_space(initial_text, 272)
        return calc_checksum(expanded_text)

    def solve_part_two(self, input_file: Optional[IO] = None) -> str:
        with input_file or self.open_input() as fp:
            initial_text = fp.read().strip()
        return calc_checksum_from_stream(fill_space(initial_text, as_iterator=True), 35651584)


def calc_checksum(text: str) -> str:
    chunk_size = 1
    checksum_len = len(text)
    while checksum_len % 2 == 0:
        chunk_size *= 2
        checksum_len //= 2

    log.debug(
        "Calculating checksum for '%s' (length=%d) with chunk size %d.", text, len(text), chunk_size
    )
    return "".join(_chunk_checksum("".join(chunk)) for chunk in create_groups(text, chunk_size))


def calc_checksum_from_stream(text: Iterator[str], length: int) -> str:
    chunk_size = 1
    checksum_len = length
    while checksum_len % 2 == 0:
        chunk_size *= 2
        checksum_len /= 2
    return "".join(
        _chunk_checksum("".join(chunk)) for chunk in create_groups(islice(text, length), chunk_size)
    )


@cache
def _chunk_checksum(chunk):
    log.debug("Calculating checksum for '%s'.", chunk)
    match chunk:
        case "00" | "11":
            return "1"
        case "01" | "10":
            return "0"

    split_at = len(chunk) // 2
    left, right = chunk[:split_at], chunk[split_at:]
    return "1" if _chunk_checksum(left) == _chunk_checksum(right) else "0"


def fill_space(
    initial_text: str, length: Optional[int] = None, as_iterator: bool = False
) -> Union[str, Iterator[str]]:
    if not as_iterator and length is None:
        raise ValueError()

    @cache
    def _interstitial(idx: int) -> str:
        if (p2 := _max_power_of_two_le(idx + 1)) == idx + 1:
            return "0"
        match _interstitial(2 * (p2 - 1) - idx):
            case "0":
                return "1"
            case "1":
                return "0"

    def _max_power_of_two_le(n):
        result = 1
        while True:
            if 2 * result > n:
                return result
            result *= 2

    reversed_inverted = "".join(("1" if c == "0" else "0") for c in initial_text[::-1])

    def _generate_chars() -> Iterator[str]:
        interstices = (_interstitial(i) for i in count())
        while True:
            yield from initial_text
            yield next(interstices)
            yield from reversed_inverted
            yield next(interstices)

    char_iter = _generate_chars()
    if as_iterator:
        return char_iter
    return "".join(islice(char_iter, length))


SAMPLE_INPUTS = [
    """\
10000
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
    initial_text = sample_input.read().strip()
    expanded_text = fill_space(initial_text, 20)
    result = calc_checksum(expanded_text)
    assert result == "01100"


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...


@pytest.mark.parametrize(
    ("text", "expected"), [("110010110100", "100"), ("10000011110010000111", "01100")]
)
def test_calc_checksum(text, expected):
    assert calc_checksum(text) == expected


@pytest.mark.parametrize(
    ("text", "length", "expected"),
    [
        ("10000", 20, "10000011110010000111"),
        ("1", 3, "100"),
        ("0", 3, "001"),
        ("1", 7, "1000110"),
        ("111100001010", 25, "1111000010100101011110000"),
        ("101", 16, "1010010010110100"),
    ],
)
def test_fill_space(text, length, expected):
    assert fill_space(text, length) == expected
