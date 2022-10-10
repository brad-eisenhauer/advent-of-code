"""Advent of Code 2019, day 15: https://adventofcode.com/2019/day/15"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(15, 2019)
        self.droid_at_o2_system: Optional[RepairDroid] = None

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        droid = RepairDroid(IntcodeMachine(program))
        result = find_oxygen_system(droid)
        self.droid_at_o2_system = result
        return result.command_count

    def solve_part_two(self) -> int:
        if self.droid_at_o2_system is None:
            self.solve_part_one()
        self.droid_at_o2_system.command_count = 0
        return fill_space_with_oxygen(self.droid_at_o2_system)


class Command(IntEnum):
    North = 1
    South = 2
    West = 3
    East = 4


class Response(IntEnum):
    Wall = 0
    Success = 1
    GoalReached = 2


@dataclass
class RepairDroid:
    brain: IntcodeMachine
    location: tuple[int, int] = (0, 0)
    command_count: int = 0

    def step(self, command: Command) -> Response:
        self.brain.input_stream = iter((command,))
        while (result := self.brain.step()) is None:
            ...
        self.command_count += 1
        result = Response(result)
        x, y = self.location
        match command, result:
            case _, Response.Wall:
                ...
            case Command.North, _:
                self.location = x, y + 1
            case Command.South, _:
                self.location = x, y - 1
            case Command.West, _:
                self.location = x - 1, y
            case Command.East, _:
                self.location = x + 1, y
        return Response(result)

    def duplicate(self) -> RepairDroid:
        machine = IntcodeMachine(
            self.brain.buffer.copy(),
            pointer=self.brain.pointer,
            relative_base=self.brain.relative_base,
        )
        return RepairDroid(machine, self.location, self.command_count)


def find_oxygen_system(repair_droid: RepairDroid) -> RepairDroid:
    visited: set[tuple[int, int]] = {repair_droid.location}
    frontier: deque[RepairDroid] = deque([repair_droid])
    while frontier:
        next_droid = frontier.popleft()
        for command in Command:
            dup_droid = next_droid.duplicate()
            response = dup_droid.step(command)
            match response:
                case Response.Wall:
                    ...
                case Response.GoalReached:
                    return dup_droid
                case Response.Success:
                    if dup_droid.location not in visited:
                        visited.add(dup_droid.location)
                        frontier.append(dup_droid)


def fill_space_with_oxygen(repair_droid: RepairDroid) -> int:
    visited: set[tuple[int, int]] = {repair_droid.location}
    max_step_count = 0
    frontier: deque[RepairDroid] = deque([repair_droid])
    while frontier:
        next_droid = frontier.popleft()
        for command in Command:
            dup_droid = next_droid.duplicate()
            response = dup_droid.step(command)
            match response:
                case Response.Success:
                    if dup_droid.location not in visited:
                        max_step_count = max(max_step_count, dup_droid.command_count)
                        visited.add(dup_droid.location)
                        frontier.append(dup_droid)
                case _:
                    ...
    return max_step_count
