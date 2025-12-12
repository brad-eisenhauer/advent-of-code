"""Advent of Code 2025, day 10: https://adventofcode.com/2025/day/10"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from io import StringIO
from typing import IO, ClassVar, Iterator, Optional, Self

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import BFS, AStar

log = logging.getLogger(__name__)


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2025, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as fp:
            while (schematic := Schematic.read(fp)) is not None:
                activator = Activator(schematic.indicators, schematic.buttons)
                initial_state = tuple(False for _ in range(len(schematic.indicators)))
                result += activator.find_min_cost_to_goal(initial_state)
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        result = 0
        with input_file or self.open_input() as fp:
            while (schematic := Schematic.read(fp)) is not None:
                log.debug("Powering machine '%s'", schematic)
                activator = Joltinator(schematic.joltage_reqs, schematic.buttons)
                initial_state = tuple(0 for _ in range(len(schematic.joltage_reqs)))
                press_count = activator.find_min_cost_to_goal(initial_state)
                log.debug("Required %d presses.", press_count)
                result += press_count
        return result


@dataclass
class Schematic:
    indicators: tuple[bool, ...]
    buttons: list[tuple[int, ...]]
    joltage_reqs: tuple[int, ...]

    PATTERN: ClassVar[str] = (
        r"\[(?P<indicators>[.#]+)\] (?P<buttons>(\([0-9,]+\) )+)\{(?P<joltage>[0-9,]+)\}"
    )

    @classmethod
    def read(cls, reader: IO) -> Self | None:
        line = reader.readline().strip()
        if (match := re.match(cls.PATTERN, line)) is None:
            log.debug("No match for '%s'", line)
            return None
        groups = match.groupdict()
        log.debug("indicators='%s' buttons='%s', joltage='%s'", *groups.values())
        indicators = tuple(c == "#" for c in groups["indicators"])
        buttons = [
            t if isinstance(t := eval(s), tuple) else (t,) for s in groups["buttons"].split()
        ]
        joltage_reqs = eval(groups["joltage"])
        return cls(indicators, buttons, joltage_reqs)


class Activator(BFS[tuple[bool, ...]]):
    def __init__(self, indicators: tuple[bool, ...], buttons: list[tuple[int, ...]]) -> None:
        self.goal_state = indicators
        self.buttons = buttons

    def generate_next_states(
        self, state: tuple[bool, ...]
    ) -> Iterator[tuple[int, tuple[bool, ...]]]:
        for button in self.buttons:
            next_state = tuple(not lit if i in button else lit for i, lit in enumerate(state))
            yield 1, next_state

    def is_goal_state(self, state: tuple[bool, ...]) -> bool:
        return state == self.goal_state


class Joltinator(AStar[tuple[int, ...]]):
    def __init__(self, joltage_reqs: tuple[int, ...], buttons: list[tuple[int, ...]]) -> None:
        self.goal_state = joltage_reqs
        self.buttons = buttons

    def generate_next_states(self, state: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
        if any(a > b for a, b in zip(state, self.goal_state)):
            return
        deficits = tuple(b - a for a, b in zip(state, self.goal_state))
        for button in self.buttons:
            min_deficit = min(d for i, d in enumerate(deficits) if i in button)
            for n in range(min_deficit):
                next_state = tuple(
                    joltage + n + 1 if i in button else joltage for i, joltage in enumerate(state)
                )
                yield n + 1, next_state

    def is_goal_state(self, state: tuple[int, ...]) -> bool:
        return state == self.goal_state

    def heuristic(self, state: tuple[int, ...]) -> int:
        """Minimum theoretical presses to goal state."""
        max_deficit = max(b - a for a, b in zip(state, self.goal_state))
        total_deficit = sum(b - a for a, b in zip(state, self.goal_state))
        max_button = max(len(b) for b in self.buttons)
        min_presses = (total_deficit + max_button - 1) // max_button
        return max(max_deficit, min_presses)


SAMPLE_INPUTS = [
    """\
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 7


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 33


class TestSchematic:
    def test_read(self, sample_input: IO) -> None:
        expected = [
            Schematic(
                (False, True, True, False),
                [(3,), (1, 3), (2,), (2, 3), (0, 2), (0, 1)],
                (3, 5, 4, 7),
            ),
            Schematic(
                (False, False, False, True, False),
                [(0, 2, 3, 4), (2, 3), (0, 4), (0, 1, 2), (1, 2, 3, 4)],
                (7, 5, 12, 7, 2),
            ),
            Schematic(
                (False, True, True, True, False, True),
                [(0, 1, 2, 3, 4), (0, 3, 4), (0, 1, 2, 4, 5), (1, 2)],
                (10, 11, 11, 5, 10, 5),
            ),
        ]
        result = []
        while (schematic := Schematic.read(sample_input)) is not None:
            result.append(schematic)
        assert result == expected
