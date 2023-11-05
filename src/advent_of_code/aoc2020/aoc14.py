"""Advent of Code 2020, day 14: https://adventofcode.com/2020/day/14"""
import re
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(14, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            result = initialize_v1(f)
        return sum(result.values())

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            result = initialize_v2(f)
        return sum(result.values())


def read_mask_v1(line: str) -> tuple[int, int]:
    value = line.split()[-1]
    and_val = int(value.replace("X", "1"), 2)
    or_val = int(value.replace("X", "0"), 2)
    return and_val, or_val


def initialize_v1(f: TextIO) -> dict[str, int]:
    result = {}
    for line in f:
        if line.startswith("mask"):
            and_val, or_val = read_mask_v1(line)
            continue
        label, value = line.split(" = ")
        value = int(value)
        value = (value & and_val) | or_val
        result[label] = value
    return result


def apply_mask_v2(mask: str, value: int) -> Iterator[int]:
    # overwrite 1s
    overwrite_mask = int(mask.replace("X", "0"), 2)
    overwrite_val = value | overwrite_mask

    floating_mask = mask.replace("0", "F").replace("1", "F")

    def generate_masks(m: str) -> Iterator[str]:
        if "X" in m:
            yield from generate_masks(m.replace("X", "0", 1))
            yield from generate_masks(m.replace("X", "1", 1))
        else:
            yield m

    for m in generate_masks(floating_mask):
        # force 1s
        result = overwrite_val | int(m.replace("F", "0"), 2)
        # force 0s
        result &= int(m.replace("F", "1"), 2)
        yield result


def initialize_v2(f: TextIO) -> dict[int, int]:
    value_pattern = re.compile(r"\[(\d+)] = (\d+)")
    result: dict[int, int] = {}
    for line in f:
        if line.startswith("mask"):
            mask = line.split(" = ")[-1].strip()
            continue
        address, value = (int(n) for n in value_pattern.search(line).groups())
        for masked_address in apply_mask_v2(mask, address):
            result[masked_address] = value
    return result


SAMPLE_INPUTS = [
    """\
mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X
mem[8] = 11
mem[7] = 101
mem[8] = 0
""",
    """\
mask = 000000000000000000000000000000X1001X
mem[42] = 100
mask = 00000000000000000000000000000000X0XX
mem[26] = 1
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[request.param]) as f:
        yield f


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_read_mask_v1(sample_input):
    result = read_mask_v1(sample_input.readline().strip())
    assert result == (2**36 - 3, 64)


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_initialize_v1(sample_input):
    result = initialize_v1(sample_input)
    assert result == {"mem[7]": 101, "mem[8]": 64}


def test_apply_mask_v2():
    mask = "000000000000000000000000000000X1001X"
    value = 42
    result = list(apply_mask_v2(mask, value))
    assert result == [26, 27, 58, 59]
