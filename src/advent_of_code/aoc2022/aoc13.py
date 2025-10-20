"""Advent of Code 2022, day 13: https://adventofcode.com/2022/day/13"""

from __future__ import annotations

import logging
import operator
from ast import literal_eval
from functools import reduce
from io import StringIO
from typing import Iterable, Iterator, Sequence, TextIO, Union

import pytest

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return sum(find_correct_packets(f))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return calc_decode_key(read_all_packets(f))


PacketElement = Union[int, list["PacketElement"]]
Packet = list[PacketElement]
MARKERS: Sequence[Packet] = ([[2]], [[6]])


def find_correct_packets(f: TextIO) -> Iterator[int]:
    index = 1
    while packets := read_packet_pair(f):
        if compare_elements(*packets) <= 0:
            yield index
        index += 1


def read_packet_pair(f: TextIO) -> tuple[Packet, ...]:
    result = []
    while (line := f.readline().rstrip()) != "":
        result.append(literal_eval(line))
    return tuple(result)


def read_all_packets(f: TextIO) -> Iterator[Packet]:
    for line in f:
        if not line.rstrip():
            continue
        yield literal_eval(line)


def compare_elements(left: PacketElement, right: PacketElement) -> int:
    match left, right:
        case lf, rt if isinstance(lf, int) and isinstance(rt, int):
            return lf - rt
        case [[], []]:
            return 0
        case [[], _]:
            return -1
        case _, []:
            return 1
        case lf, rt:
            if not isinstance(lf, list):
                lf = [lf]
            if not isinstance(rt, list):
                rt = [rt]
            result = compare_elements(lf[0], rt[0])
            if result == 0:
                result = compare_elements(lf[1:], rt[1:])
            return result


def calc_decode_key(packets: Iterable[Packet], markers: Sequence[Packet] = MARKERS) -> int:
    marker_indexes = [i + 1 for i in range(len(markers))]
    for packet in packets:
        for i, marker in enumerate(markers):
            if compare_elements(packet, marker) < 0:
                # assumes markers are sorted ascending
                for j in range(i, len(marker_indexes)):
                    marker_indexes[j] += 1
                break
    return reduce(operator.mul, marker_indexes)


SAMPLE_INPUTS = [
    """\
[1,1,3,1,1]
[1,1,5,1,1]

[[1],[2,3,4]]
[[1],4]

[9]
[[8,7,6]]

[[4,4],4,4]
[[4,4],4,4,4]

[7,7,7,7]
[7,7,7]

[]
[3]

[[[]]]
[[]]

[1,[2,[3,[4,[5,6,7]]]],8,9]
[1,[2,[3,[4,[5,6,0]]]],8,9]
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_packet_from_str(sample_input):
    line = sample_input.readline()
    result = literal_eval(line)
    assert result == [1, 1, 3, 1, 1]
    line = sample_input.readline()
    result = literal_eval(line)
    assert result == [1, 1, 5, 1, 1]


def test_read_packet_pair(sample_input):
    assert read_packet_pair(sample_input) == ([1, 1, 3, 1, 1], [1, 1, 5, 1, 1])
    assert read_packet_pair(sample_input) == ([[1], [2, 3, 4]], [[1], 4])


def test_compare_packets(sample_input):
    left, right = read_packet_pair(sample_input)
    assert compare_elements(left, right) == -2

    left, right = read_packet_pair(sample_input)
    assert compare_elements(left, right) == -2


def test_find_correct_packets(sample_input):
    assert list(find_correct_packets(sample_input)) == [1, 2, 4, 6]


def test_read_all_packets(sample_input):
    packet_count = sum(1 for _ in read_all_packets(sample_input))
    assert packet_count == 16


def test_calc_decode_key(sample_input):
    assert calc_decode_key(read_all_packets(sample_input)) == 140
