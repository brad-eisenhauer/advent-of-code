"""Advent of Code 2024, day 9: https://adventofcode.com/2024/day/9"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(9, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            layout = [int(n) for n in reader.read().strip()]
        return calc_defragged_checksum(layout)

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            defragger = Defragger.read(reader)
        defragger.defrag()
        return defragger.calc_checksum()


@dataclass(frozen=True)
class StorageBlock:
    loc: int
    size: int
    is_empty: bool = False
    file_id: int = 0

    def calc_checksum(self, loc: int) -> int:
        return self.file_id * (loc * self.size + self.size * (self.size - 1) // 2)


@dataclass
class Defragger:
    layout: list[StorageBlock]

    @classmethod
    def read(cls, reader: IO) -> Defragger:
        result = []
        file_id = 0
        loc = 0
        for i, char in enumerate(reader.read().strip()):
            block_size = int(char)
            if not block_size:
                continue
            if i % 2:
                result.append(StorageBlock(loc, block_size, is_empty=True))
            else:
                result.append(StorageBlock(loc, block_size, file_id=file_id))
                file_id += 1
            loc += block_size
        return cls(result)

    def defrag(self) -> None:
        blocks_to_defrag = [b for b in self.layout[::-1] if not b.is_empty]
        for file_block in blocks_to_defrag:
            self.defrag_block(file_block)

    def defrag_block(self, file_block: StorageBlock) -> None:
        block_index = self._find_block_index(file_block.loc)
        log.debug("Found file_id=%d at index=%d.", file_block.file_id, block_index)
        for free_index, free_block in enumerate(self.layout[:block_index]):
            if not free_block.is_empty:
                continue
            if free_block.size == file_block.size:
                log.debug(
                    "Swapping file_id=%d from %d to free space at %d.",
                    file_block.file_id,
                    file_block.loc,
                    free_block.loc,
                )
                self.layout[block_index] = StorageBlock(
                    file_block.loc, file_block.size, is_empty=True
                )
                self.layout[free_index] = replace(file_block, loc=free_block.loc)
                break
            if free_block.size > file_block.size:
                log.debug(
                    "Swapping file_id=%d from %d to free space at %d.",
                    file_block.file_id,
                    file_block.loc,
                    free_block.loc,
                )
                self.layout[block_index] = StorageBlock(
                    file_block.loc, file_block.size, is_empty=True
                )
                self.layout[free_index] = replace(file_block, loc=free_block.loc)
                self.layout.insert(
                    free_index + 1,
                    StorageBlock(
                        free_block.loc + file_block.size,
                        free_block.size - file_block.size,
                        is_empty=True,
                    ),
                )
                break

    def _find_block_index(self, loc: int) -> int | None:
        hi = len(self.layout)
        lo = 0
        while hi >= lo:
            idx = lo + (hi - lo) // 2
            if self.layout[idx].loc == loc:
                return idx
            if self.layout[idx].loc > loc:
                hi = idx
            else:
                lo = idx + 1
        return None

    def calc_checksum(self) -> int:
        result = 0
        loc = 0
        for block in self.layout:
            result += block.calc_checksum(loc)
            loc += block.size
        return result


def calc_defragged_checksum(disk_layout: list[int]) -> int:
    head_index = 0
    head_file_index = 0
    tail_index = 2 * (len(disk_layout) // 2)
    tail_file_index = len(disk_layout) // 2
    total_blocks_filled = 0
    result = 0

    while tail_index >= head_index:
        file_blocks = disk_layout[head_index]
        log.debug(
            "Positions %d-%d contain file %d (checksum=%d)",
            total_blocks_filled,
            total_blocks_filled + file_blocks - 1,
            head_file_index,
            head_file_index * range_sum(total_blocks_filled, file_blocks),
        )
        result += head_file_index * range_sum(total_blocks_filled, file_blocks)
        total_blocks_filled += file_blocks

        free_blocks = disk_layout[head_index + 1]
        while free_blocks and tail_index > head_index:
            tail_file_blocks = disk_layout[tail_index]
            if tail_file_blocks > free_blocks:
                log.debug(
                    "Positions %d-%d contain file %d (checksum=%d)",
                    total_blocks_filled,
                    total_blocks_filled + free_blocks - 1,
                    tail_file_index,
                    tail_file_index * range_sum(total_blocks_filled, free_blocks),
                )
                result += tail_file_index * range_sum(total_blocks_filled, free_blocks)
                disk_layout[tail_index] -= free_blocks
                total_blocks_filled += free_blocks
                free_blocks = 0
            elif tail_file_blocks < free_blocks:
                log.debug(
                    "Positions %d-%d contain file %d (checksum=%d)",
                    total_blocks_filled,
                    total_blocks_filled + tail_file_blocks - 1,
                    tail_file_index,
                    tail_file_index * range_sum(total_blocks_filled, tail_file_blocks),
                )
                result += tail_file_index * range_sum(total_blocks_filled, tail_file_blocks)
                free_blocks -= tail_file_blocks
                total_blocks_filled += tail_file_blocks
                tail_index -= 2
                tail_file_index -= 1
            else:
                log.debug(
                    "Positions %d-%d contain file %d (checksum=%d)",
                    total_blocks_filled,
                    total_blocks_filled + free_blocks - 1,
                    tail_file_index,
                    tail_file_index * range_sum(total_blocks_filled, free_blocks),
                )
                result += tail_file_index * range_sum(total_blocks_filled, free_blocks)
                total_blocks_filled += free_blocks
                free_blocks = 0
                tail_index -= 2
                tail_file_index -= 1

        head_index += 2
        head_file_index += 1

    return result


def range_sum(start: int, length: int) -> int:
    return start * length + length * (length - 1) // 2


SAMPLE_INPUTS = [
    """\
12345
""",
    """\
2333133121414131402
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize("sample_input", [0], indirect=True)
def test_read_defragger(sample_input: IO) -> None:
    assert Defragger.read(sample_input) == Defragger(
        layout=[
            StorageBlock(loc=0, size=1, file_id=0),
            StorageBlock(loc=1, size=2, is_empty=True),
            StorageBlock(loc=3, size=3, file_id=1),
            StorageBlock(loc=6, size=4, is_empty=True),
            StorageBlock(loc=10, size=5, file_id=2),
        ]
    )


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_defrag(sample_input: IO) -> None:
    defragger = Defragger.read(sample_input)
    defragger.defrag()
    assert defragger.layout == [
        StorageBlock(loc=0, size=2, file_id=0),
        StorageBlock(loc=2, size=2, file_id=9),
        StorageBlock(loc=4, size=1, file_id=2),
        StorageBlock(loc=5, size=3, file_id=1),
        StorageBlock(loc=8, size=3, file_id=7),
        StorageBlock(loc=11, size=1, is_empty=True),
        StorageBlock(loc=12, size=2, file_id=4),
        StorageBlock(loc=14, size=1, is_empty=True),
        StorageBlock(loc=15, size=3, file_id=3),
        StorageBlock(loc=18, size=4, is_empty=True),
        StorageBlock(loc=22, size=4, file_id=5),
        StorageBlock(loc=26, size=1, is_empty=True),
        StorageBlock(loc=27, size=4, file_id=6),
        StorageBlock(loc=31, size=5, is_empty=True),
        StorageBlock(loc=36, size=4, file_id=8),
        StorageBlock(loc=40, size=2, is_empty=True),
    ]


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_defrag_block(sample_input: IO) -> None:
    defragger = Defragger.read(sample_input)
    file_block = defragger.layout[-1]
    defragger.defrag_block(file_block)
    assert defragger.layout == [
        StorageBlock(loc=0, size=2, file_id=0),
        StorageBlock(loc=2, size=2, file_id=9),
        StorageBlock(loc=4, size=1, is_empty=True),
        StorageBlock(loc=5, size=3, file_id=1),
        StorageBlock(loc=8, size=3, is_empty=True),
        StorageBlock(loc=11, size=1, file_id=2),
        StorageBlock(loc=12, size=3, is_empty=True),
        StorageBlock(loc=15, size=3, file_id=3),
        StorageBlock(loc=18, size=1, is_empty=True),
        StorageBlock(loc=19, size=2, file_id=4),
        StorageBlock(loc=21, size=1, is_empty=True),
        StorageBlock(loc=22, size=4, file_id=5),
        StorageBlock(loc=26, size=1, is_empty=True),
        StorageBlock(loc=27, size=4, file_id=6),
        StorageBlock(loc=31, size=1, is_empty=True),
        StorageBlock(loc=32, size=3, file_id=7),
        StorageBlock(loc=35, size=1, is_empty=True),
        StorageBlock(loc=36, size=4, file_id=8),
        StorageBlock(loc=40, size=2, is_empty=True),
    ]


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 60), (1, 1928)], indirect=["sample_input"]
)
def test_part_one(solution: AocSolution, sample_input: IO, expected: int) -> None:
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize("sample_input", [1], indirect=True)
def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 2858
