"""Advent of Code 2025, day 12: https://adventofcode.com/2025/day/12"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache, cached_property
from io import StringIO
from itertools import product, repeat
import re
from typing import IO, Iterator, Optional, Self, TypeAlias

import pytest

from advent_of_code.base import Solution

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        presents: dict[int, Present] = {}
        result = 0
        with input_file or self.open_input() as fp:
            for item in read(fp):
                if item is None:
                    break
                if isinstance(item, Present):
                    presents[item.id] = item
                    continue
                region, present_counts = item
                present_list: list[Present] = []
                for id, count in enumerate(present_counts):
                    present_list.extend(repeat(presents[id], count))
                if region.can_place(present_list):
                    result += 1
        return result


    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            ...


Vector: TypeAlias = tuple[int, int]


@dataclass(frozen=True)
class Present:
    id: int
    shape: tuple[Vector]

    @classmethod
    def read(cls, id: int, reader: IO) -> Self:
        coords = []
        for row, line in enumerate(reader):
            if not line.strip():
                break
            for col, char in enumerate(line):
                if char == "#":
                    coords.append((row, col))
        return cls(id, tuple(coords))

    @property
    def normalized(self) -> Self:
        row_offset = -min(row for row, _ in self.shape)
        col_offset = -min(col for _, col in self.shape)
        return self.translate(row_offset, col_offset)

    @cached_property
    def rotated(self) -> Self:
        return Present(self.id, tuple((-col, row) for row, col in self.shape)).normalized

    @cache
    def translate(self, row_offset: int, col_offset: int) -> Self:
        return Present(self.id, tuple((row + row_offset, col + col_offset) for row, col in self.shape))

    def generate_rotations(self) -> Iterator[Present]:
        yield self
        next_rotation = self.rotated
        while next_rotation.normalized != self.normalized:
            yield next_rotation
            next_rotation = next_rotation.rotated

    @property
    def dims(self) -> Vector:
        max_row = max(row for row, _ in self.normalized)
        max_col = max(col for _, col in self.normalized)
        return max_row + 1, max_col + 1

    def __len__(self) -> int:
        return len(self.shape)

    def __iter__(self) -> Iterator[Vector]:
        return iter(self.shape)


@dataclass(frozen=True)
class Region:
    coords: tuple[Vector]

    @classmethod
    def from_dimensions(cls, dims: Vector) -> Self:
        rows, cols = dims
        return cls(tuple(product(range(rows), range(cols))))

    @cached_property
    def bbox(self) -> tuple[Vector, Vector]:
        min_row = min(row for row, _ in self.coords)
        min_col = min(col for _, col in self.coords)
        max_row = max(row for row, _ in self.coords)
        max_col = max(col for _, col in self.coords)
        return ((min_row, min_col), (max_row, max_col))

    @property
    def dims(self) -> Vector:
        (min_row, min_col), (max_row, max_col) = self.bbox
        return max_row - min_row + 1, max_col - min_col + 1

    def place_present(self, present: Present) -> Region | None:
        log.debug("Attempting to place %s in %s.", present, self)
        if any(c not in self.coords for c in present.shape):
            return None
        return Region(tuple(c for c in self.coords if c not in present))

    def can_place(self, presents: list[Present]) -> bool:
        reg_width, reg_height = self.dims
        reg_width //= 3
        reg_height //= 3
        if len(presents) <= reg_width * reg_height:
            return True
        if sum(len(p) for p in presents) > len(self.coords):
            return False

        raise ValueError("Too complicated past this point.")

        next_present, *remaining_presents = presents
        (min_row, min_col), (max_row, max_col) = self.bbox
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Testing/Region:")
            for row in range(min_row, max_row + 1):
                line = "".join("#" if (row, col) in self.coords else "." for col in range(min_col, max_col + 1))
                log.debug(f"\t{line}")
            log.debug("Testing/Shape:")
            for row in range(next_present.dims[0]):
                line = "".join("#" if (row, col) in next_present.shape else "." for col in range(next_present.dims[1]))
                log.debug(f"\t{line}")

        # Try every rotation and translation of the next present that fits into the bounding box.
        for rotated_present in next_present.generate_rotations():
            for row_offset, col_offset in product(
                range(min_row, max_row - rotated_present.dims[0] + 1),
                range(min_col, max_col - rotated_present.dims[1] + 1),
            ):
                if (reduced_region := self.place_present(rotated_present.translate(row_offset, col_offset))) is None:
                    continue
                log.debug("Placement found.")
                if reduced_region.can_place(remaining_presents):
                    return True
        log.debug("No solution found.")
        return False


def read(reader: IO) -> Iterator[Present | tuple[Region, list[int]] | None]:
    SHAPE_ID_PATTERN = r"^(?P<shape_id>[0-9])+:$"
    REGION_PATTERN = r"^(?P<cols>[0-9]+)x(?P<rows>[0-9]+):(?P<present_counts>( [0-9]+)+)$"

    while True:
        if not (line := reader.readline().strip()):
            break
        if (match := re.match(SHAPE_ID_PATTERN, line)) is not None:
            shape_id = int(match.groupdict()["shape_id"])
            yield Present.read(shape_id, reader)
            continue
        if (match := re.match(REGION_PATTERN, line)) is not None:
            rows = int(match.groupdict()["rows"])
            cols = int(match.groupdict()["cols"])
            present_counts = list(int(n) for n in match.groupdict()["present_counts"].split())
            yield Region.from_dimensions((rows, cols)), present_counts
            continue
        log.warning("'%s' did not match any pattern.", line)


SAMPLE_INPUTS = [
    """\
0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
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
    assert solution.solve_part_one(sample_input) == 2


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...


@pytest.fixture()
def present() -> Present:
    return Present(1, ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)))


class TestPresent:
    def test_read(self, present: Present) -> None:
        reader = StringIO("#..\n#..\n###\n")
        actual = Present.read(1, reader)
        assert actual == present

    def test_rotate(self, present: Present) -> None:
        assert present.rotated == Present(1, ((0, 0), (0, 1), (0, 2), (-1, 2), (-2, 2)))

    def test_normalize(self, present: Present) -> None:
        assert present.normalized == present
        assert present.rotated.normalized == Present(1, ((2, 0), (2, 1), (2, 2), (1, 2), (0, 2)))

    def test_generate_rotations(self, present: Present) -> None:
        rotations = list(present.generate_rotations())
        expected = [
            present,
            Present(1, ((0, 0), (0, 1), (0, 2), (-1, 2), (-2, 2))),
            Present(1, ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, -2))),
            Present(1, ((0, 0), (0, -1), (0, -2), (1, -2), (2, -2))),
        ]
        assert rotations == expected
