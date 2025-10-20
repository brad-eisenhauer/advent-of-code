"""Advent of Code 2016, day 14: https://adventofcode.com/2016/day/14"""

from __future__ import annotations

import re
from collections import deque
from hashlib import md5
from io import StringIO
from itertools import count, islice
from typing import Iterator

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            salt = fp.read().strip()
        otp = OTP(generate_hashes(salt))
        *_, (index, _) = islice(otp, 64)
        return index

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            salt = fp.read().strip()
        otp = OTP(generate_hashes(salt, stretch_factor=2017))
        *_, (index, _) = islice(otp, 64)
        return index


def generate_hashes(salt: str, stretch_factor: int = 1) -> Iterator[tuple[int, str]]:
    for n in count():
        result = f"{salt}{n}"
        for _ in range(stretch_factor):
            result = md5(result.encode()).hexdigest()
        yield n, result


class OTP(Iterator[tuple[int, str]]):
    _candidate_pattern = r"((.)\2{2,})"

    def __init__(self, hash_generator: Iterator[tuple[int, str]]):
        self._hash_generator = hash_generator
        self._candidate_hashes: deque[tuple[int, str, list[str]]] = deque()

    def __next__(self) -> tuple[int, str]:
        for index, hash, patterns in self._get_candidate_hashes():
            if self._confirm(index, patterns[0][0]):
                return index, hash

    def _get_candidate_hashes(self) -> Iterator[tuple[int, str, list[str]]]:
        while True:
            if self._candidate_hashes:
                yield self._candidate_hashes.popleft()
            else:
                yield self._find_next_candidate()

    def _find_next_candidate(self) -> tuple[int, str, list[str]]:
        for index, hash in self._hash_generator:
            if patterns := self._find_pattens(hash):
                return index, hash, patterns

    def _find_pattens(self, hash: str) -> set(str):
        return [m[0] for m in re.findall(self._candidate_pattern, hash)]

    def _confirm(self, from_index: int, char: str) -> bool:
        for index, hash, patterns in self._peek_queue():
            if index - from_index > 1000:
                return False
            if any(char in p and len(p) >= 5 for p in patterns):
                log.debug("Hash at index %d confirmed by %s at index %d", from_index, hash, index)
                return True

    def _peek_queue(self) -> Iterator[tuple[int, str, list[str]]]:
        yield from self._candidate_hashes
        while True:
            result = self._find_next_candidate()
            self._candidate_hashes.append(result)
            yield result


SAMPLE_INPUTS = [
    """\
abc""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture
def otp(sample_input, request):
    salt = sample_input.read().strip()
    stretch_factor = 1 if hasattr(request, "param") else request.param["stretch_factor"]
    return OTP(generate_hashes(salt, stretch_factor=stretch_factor))


@pytest.mark.parametrize(
    ("otp", "expected"),
    [({"stretch_factor": 1}, 22728), ({"stretch_factor": 2017}, 22859)],
    indirect=["otp"],
)
def test_64th_hash_index(otp, expected):
    *_, (index, _) = islice(otp, 64)
    assert index == expected


@pytest.mark.parametrize(
    ("otp", "expected"),
    [({"stretch_factor": 1}, 39), ({"stretch_factor": 2017}, 10)],
    indirect=["otp"],
)
def test_first_hash_index(otp, expected):
    index, _ = next(otp)
    assert index == expected


def test_second_hash_index(otp):
    *_, (index, _) = islice(otp, 2)
    assert index == 92
