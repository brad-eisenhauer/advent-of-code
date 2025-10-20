from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution
from advent_of_code.util.math import least_common_multiple


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            array = MoonArray.read_array(f)
        for _ in range(1000):
            array = array.step()
        return array.total_energy()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            array = MoonArray.read_array(f)
        cycles = (ax.steps_to_cycle() for ax in array.axes)
        return reduce(least_common_multiple, cycles)


@dataclass(frozen=True)
class MoonAxis:
    positions: tuple[int, ...]
    velocities: tuple[int, ...]

    @classmethod
    def from_positions(cls, positions: tuple[int, ...]) -> MoonAxis:
        velocities = (0,) * len(positions)
        return cls(positions, velocities)

    def acceleration(self) -> tuple[int, ...]:
        less_than_count = (sum(1 for p2 in self.positions if p2 < p1) for p1 in self.positions)
        greater_than_count = (sum(1 for p2 in self.positions if p2 > p1) for p1 in self.positions)
        return tuple(b - a for a, b in zip(less_than_count, greater_than_count))

    def step(self) -> MoonAxis:
        new_velos = tuple(v + a for v, a in zip(self.velocities, self.acceleration()))
        new_pos = tuple(p + v for p, v in zip(self.positions, new_velos))
        return MoonAxis(new_pos, new_velos)

    def steps_to_cycle(self) -> int:
        axis = self.step()
        step_count = 1
        while axis != self:
            axis = axis.step()
            step_count += 1
        return step_count


@dataclass(frozen=True)
class MoonArray:
    axes: tuple[MoonAxis, ...]

    @classmethod
    def read_array(cls, f: TextIO) -> MoonArray:
        pattern = re.compile(r"^<x=(-?\d+), y=(-?\d+), z=(-?\d+)>")
        positions = (tuple(int(n) for n in pattern.match(line).groups()) for line in f.readlines())
        axes = tuple(MoonAxis.from_positions(ps) for ps in zip(*positions))
        return cls(axes)

    def step(self) -> MoonArray:
        axes = tuple(ax.step() for ax in self.axes)
        return MoonArray(axes)

    def total_energy(self) -> int:
        potential_energies = [
            sum(abs(p) for p in ps) for ps in zip(*(ax.positions for ax in self.axes))
        ]
        kinetic_energies = [
            sum(abs(v) for v in vs) for vs in zip(*(ax.velocities for ax in self.axes))
        ]
        return sum(p * k for p, k in zip(potential_energies, kinetic_energies))


SAMPLE_INPUT = """\
<x=-1, y=0, z=2>
<x=2, y=-10, z=-7>
<x=4, y=-8, z=8>
<x=3, y=5, z=-1>
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        return MoonArray.read_array(f)


class TestMoonAxis:
    @pytest.mark.parametrize(
        ("axis", "expected"),
        [
            (MoonAxis.from_positions((-1, 2, 4, 3)), MoonAxis((2, 3, 1, 2), (3, 1, -3, -1))),
            (MoonAxis.from_positions((0, -10, -8, 5)), MoonAxis((-1, -7, -7, 2), (-1, 3, 1, -3))),
            (MoonAxis.from_positions((2, -7, 8, -1)), MoonAxis((1, -4, 5, 0), (-1, 3, -3, 1))),
        ],
    )
    def test_step(self, axis, expected):
        assert axis.step() == expected


class TestMoonArray:
    def test_read_array(self, sample_input):
        expected = MoonArray(
            (
                MoonAxis.from_positions((-1, 2, 4, 3)),
                MoonAxis.from_positions((0, -10, -8, 5)),
                MoonAxis.from_positions((2, -7, 8, -1)),
            )
        )
        assert sample_input == expected

    @pytest.mark.parametrize(
        ("step_count", "expected"),
        [
            (
                1,
                MoonArray(
                    (
                        MoonAxis((2, 3, 1, 2), (3, 1, -3, -1)),
                        MoonAxis((-1, -7, -7, 2), (-1, 3, 1, -3)),
                        MoonAxis((1, -4, 5, 0), (-1, 3, -3, 1)),
                    )
                ),
            ),
            (
                2,
                MoonArray(
                    (
                        MoonAxis((5, 1, 1, 1), (3, -2, 0, -1)),
                        MoonAxis((-3, -2, -4, -4), (-2, 5, 3, -6)),
                        MoonAxis((-1, 2, -1, 2), (-2, 6, -6, 2)),
                    )
                ),
            ),
            (
                10,
                MoonArray(
                    (
                        MoonAxis((2, 1, 3, 2), (-3, -1, 3, 1)),
                        MoonAxis((1, -8, -6, 0), (-2, 1, 2, -1)),
                        MoonAxis((-3, 0, 1, 4), (1, 3, -3, -1)),
                    )
                ),
            ),
        ],
    )
    def test_stepwise(self, sample_input, step_count, expected):
        array = sample_input
        for _ in range(step_count):
            array = array.step()
        assert array == expected

    def test_total_energy(self, sample_input):
        array = sample_input
        for _ in range(10):
            array = array.step()
        assert array.total_energy() == 179
