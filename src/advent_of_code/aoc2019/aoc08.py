"""Advent of Code 2019, Day 08: https://adventofcode.com/2019/day/8"""

import sys
from collections import Counter
from io import StringIO
from typing import Iterator, TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.ocr import DEFAULT_BLOCK, PrintToString


class AocSolution(Solution[int, str]):
    def __init__(self, **kwargs):
        super().__init__(8, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            data = f.readline().strip()
        image = SifImage((25, 6), data)

        min_zeros_counts = None
        for layer in image.layers():
            digit_counts = Counter(layer)
            if min_zeros_counts is None or digit_counts["0"] < min_zeros_counts["0"]:
                min_zeros_counts = digit_counts

        return min_zeros_counts["1"] * min_zeros_counts["2"]

    def solve_part_two(self) -> str:
        with self.open_input() as f:
            data = f.readline().strip()
        image = SifImage((25, 6), data)
        with (converter := PrintToString()) as out_file:
            image.print(out_file)
        return converter.to_string()


class SifImage:
    BLOCK = DEFAULT_BLOCK

    def __init__(self, dims: tuple[int, int], data: str):
        self.dims = dims
        self.data = data

    def layers(self) -> Iterator[str]:
        width, height = self.dims
        step = width * height
        index = 0
        while index < len(self.data):
            yield self.data[index : index + step]
            index += step

    def render(self) -> str:
        *layers, base = self.layers()
        for layer in reversed(layers):
            base = "".join(b if l == "2" else l for l, b in zip(layer, base))
        return base

    def print(self, f: TextIO = sys.stdout):
        printable = self.render().replace("0", " ").replace("1", self.BLOCK)
        width, _ = self.dims
        for index in range(0, len(printable), width):
            f.write(printable[index : index + width] + "\n")


SAMPLE_INPUT = """\
123456789012
"""


@pytest.fixture
def sample_input() -> TextIO:
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_layers(sample_input):
    data = sample_input.readline().strip()
    dims = (3, 2)
    image = SifImage(dims, data)
    layers = list(image.layers())
    assert len(layers) == 2
    assert all(len(layer) == 6 for layer in layers)


def test_render():
    data = "0222112222120000"
    image = SifImage((2, 2), data)
    assert image.render() == "0110"
