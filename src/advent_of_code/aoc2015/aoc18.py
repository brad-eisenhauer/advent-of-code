"""Advent of Code 2015, day 18: https://adventofcode.com/2015/day/18"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import product
from typing import IO

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(18, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            grid = LightGrid.read(f)
        for _ in range(100):
            grid = grid.step()
        return len(grid.lights)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            grid = LightGrid.read(f, stuck_corners=True)
        for _ in range(100):
            grid = grid.step()
        return len(grid.lights)


@dataclass(frozen=True)
class LightGrid:
    lights: set[tuple[int, int]]
    dims: tuple[int, int] = (100, 100)
    stuck_corners: set[tuple[int, int]] = field(default_factory=set)

    @classmethod
    def read(cls, f: IO, stuck_corners: bool = False) -> LightGrid:
        lights = set()
        for y, line in enumerate(f):
            for x, c in enumerate(line.strip()):
                if c == "#":
                    lights.add((x, y))

        corners = {(0, 0), (0, y), (x, 0), (x, y)} if stuck_corners else set()

        return LightGrid(lights, (x + 1, y + 1), corners)

    def step(self):
        neighbor_count = self.count_neighbors()
        next_lights = {light for light in self.lights if neighbor_count[light] in (2, 3)}
        next_lights |= {
            light
            for light in neighbor_count
            if light not in self.lights
            and neighbor_count[light] == 3
            and self.light_is_valid(light)
        }
        next_lights |= self.stuck_corners
        return replace(self, lights=next_lights)

    def count_neighbors(self) -> dict[tuple[int, int], int]:
        neighbor_count = defaultdict(int)
        for light in self.lights:
            for offset in product((1, 0, -1), (1, 0, -1)):
                if offset == (0, 0):
                    continue
                neighbor = tuple(sum(xs) for xs in zip(light, offset))
                neighbor_count[neighbor] += 1
        return neighbor_count

    def light_is_valid(self, light: tuple[int, int]) -> bool:
        return all(val in range(limit) for val, limit in zip(light, self.dims))
