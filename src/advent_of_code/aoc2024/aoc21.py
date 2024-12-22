"""Advent of Code 2024, day 21: https://adventofcode.com/2024/day/21"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from io import StringIO
from itertools import pairwise, product
from typing import IO, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution

Key: TypeAlias = str
"""A key on a keypad; must be a single-character."""
KeyLocation: TypeAlias = complex
"""The location of a key on a keypad."""
Command: TypeAlias = str
"""A series of key presses to navigate to and activate a button.

A command will always contain exactly one "A" which will be at its end.
"""
CommandSequence: TypeAlias = list[Command]
"""A sequence of commands to be entered on a keypad."""


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        keypads = (NUMERIC_KEYPAD,) + (DIRECTIONAL_KEYPAD,) * 2
        result = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                line = line.strip()
                value = int(line[:-1])
                commands = count_command_occurrences(keypads, line)
                result += value * sum(len(command) * count for command, count in commands.items())
        return result

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        keypads = (NUMERIC_KEYPAD,) + (DIRECTIONAL_KEYPAD,) * 25
        result = 0
        with input_file or self.open_input() as reader:
            for line in reader:
                line = line.strip()
                value = int(line[:-1])
                commands = count_command_occurrences(keypads, line)
                result += value * sum(len(command) * count for command, count in commands.items())
        return result


@cache
def count_command_occurrences(
    keypads: tuple[Keypad], target_command: Command
) -> dict[Command, int]:
    """Count the occurrence of commands in successor sequences to produce the target command.

    The initial position is always presumed to be "A".

    Parameters
    ----------
    keypads: tuple[Keypad]
        Series of keypads. target_command is to be entered on the first keypad. Function counts the
        occurrence of commands entered into the last keypad.
    target_command: Command
        Command to be entered into the first keypad.

    Returns
    -------
    dict[Command, int]
        The count of each command entered into the last keypad to produce the target command on the
        first keypad.
    """
    keypad = keypads[0]
    command_counts: dict[Command, int] = defaultdict(int)
    for origin, target in pairwise("A" + target_command):
        candidate_commands = keypad.navigate(origin, target)
        if len(candidate_commands) > 1:
            command = min(candidate_commands, key=calc_min_successor_command_sequence_length)
        else:
            command = candidate_commands[0]
        command_counts[command] += 1

    if len(keypads) == 1:
        return command_counts

    result: dict[Command, int] = defaultdict(int)
    for command, count in command_counts.items():
        next_command_counts = count_command_occurrences(keypads[1:], command)
        for next_command, next_command_count in next_command_counts.items():
            result[next_command] += count * next_command_count

    return result


def calc_min_successor_command_sequence_length(command: Command, depth: int = 4) -> int:
    """Calculate the minimum expansion of the given command through `depth` directional keypads."""
    candidate_commands: list[list[Command]] = [
        DIRECTIONAL_KEYPAD.navigate(origin, target) for origin, target in pairwise("A" + command)
    ]
    candidate_command_sequences: list[CommandSequence] = list(product(*candidate_commands))
    if depth <= 1:
        return min(
            sum(len(part) for part in candidate) for candidate in candidate_command_sequences
        )
    return min(
        sum(
            calc_min_successor_command_sequence_length(command, depth - 1)
            for command in command_seq
        )
        for command_seq in candidate_command_sequences
    )


@dataclass
class Keypad:
    keys: dict[Key, KeyLocation]

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        self.values = set(self.keys.values())

    @cache
    def navigate(self, origin: Key, target: Key) -> list[Command]:
        """Returns a list of 1-2 possible commands to navigate to and activate the target key."""
        origin_loc = self.keys[origin]
        target_loc = self.keys[target]
        diff = target_loc - origin_loc
        corner_locs = [origin_loc + diff.real, origin_loc + diff.imag * 1j]
        horiz_part = (">" if diff.real > 0 else "<") * int(abs(diff.real))
        vert_part = ("^" if diff.imag > 0 else "v") * int(abs(diff.imag))
        key_sequences = [horiz_part + vert_part, vert_part + horiz_part]
        valid_sequences = set(
            seq for seq, corner in zip(key_sequences, corner_locs) if corner in self.values
        )
        return list(seq + "A" for seq in valid_sequences)


NUMERIC_KEYPAD = Keypad(
    keys={
        "0": 1,
        "A": 2,
        "1": 1j,
        "2": 1 + 1j,
        "3": 2 + 1j,
        "4": 2j,
        "5": 1 + 2j,
        "6": 2 + 2j,
        "7": 3j,
        "8": 1 + 3j,
        "9": 2 + 3j,
    }
)
DIRECTIONAL_KEYPAD = Keypad(keys={"<": 0, "v": 1, ">": 2, "^": 1 + 1j, "A": 2 + 1j})


SAMPLE_INPUTS = [
    """\
029A
980A
179A
456A
379A
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 126384
