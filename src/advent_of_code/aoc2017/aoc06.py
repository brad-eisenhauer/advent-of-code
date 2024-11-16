"""Advent of Code 2017, day 6: https://adventofcode.com/2017/day/6"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import IO, Iterator, Optional
from typing_extensions import Self

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util.cycle import detect_cycle


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(6, 2017, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            initial_state = State.read(fp)

        def _runner() -> Iterator[State]:
            state = initial_state
            while True:
                yield state
                state = state.step()

        cycle_len, result = detect_cycle(_runner())
        log.debug("Detected cycle length: %d", cycle_len)
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            initial_state = State.read(fp)

        def _runner() -> Iterator[State]:
            state = initial_state
            while True:
                yield state
                state = state.step()

        cycle_len, _ = detect_cycle(_runner())
        log.debug("Detected cycle length: %d", cycle_len)
        return cycle_len


@dataclass(frozen=True)
class State:
    block_counts: tuple[int, ...]

    @classmethod
    def read(cls, reader: IO) -> Self:
        block_counts = tuple(int(n) for n in reader.readline().split())
        log.debug("Blocks: %s", block_counts)
        return cls(block_counts=block_counts)

    def step(self) -> Self:
        max_blocks = max(self.block_counts)
        max_blocks_index = self.block_counts.index(max_blocks)
        new_block_counts = list(self.block_counts)
        new_block_counts[max_blocks_index] = 0
        new_block_counts = [n + max_blocks // len(self.block_counts) for n in new_block_counts]
        for i in range(max_blocks % len(self.block_counts)):
            new_block_counts[(i + max_blocks_index + 1) % len(self.block_counts)] += 1
        return type(self)(block_counts=tuple(new_block_counts))


SAMPLE_INPUTS = [
    """\
0\t2\t7\t0
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def state(sample_input: IO) -> State:
    return State.read(sample_input)


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 5


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == ...


class TestState:
    def test_step(self, state: State) -> None:
        assert state.step() == State(block_counts=(2, 4, 1, 2))
