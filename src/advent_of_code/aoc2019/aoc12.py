from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from typing import TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(12, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            array = MoonArray.read_array(f)
        for _ in range(1000):
            array = array.step()
        return array.total_energy()

    def solve_part_two(self) -> int:
        step_count = 0
        prior_states = set()
        with self.open_input() as f:
            array = MoonArray.read_array(f)
        while array not in prior_states:
            prior_states.add(array)
            array = array.step()
            step_count += 1
        return step_count


@dataclass(frozen=True)
class Moon:
    position: tuple[int, ...]
    velocity: tuple[int, ...] = (0, 0, 0)

    def apply_velocity_delta(self, delta: tuple[int, ...]) -> Moon:
        return Moon(self.position, tuple(a + b for a, b in zip(self.velocity, delta)))

    def advance(self) -> Moon:
        return Moon(tuple(a + b for a, b in zip(self.position, self.velocity)), self.velocity)

    def total_energy(self) -> int:
        return sum(abs(n) for n in self.position) * sum(abs(n) for n in self.velocity)


@dataclass(frozen=True)
class MoonArray:
    moons: frozenset[Moon]

    @classmethod
    def read_array(cls, f: TextIO) -> MoonArray:
        pattern = re.compile(r"^<x=(-?\d+), y=(-?\d+), z=(-?\d+)>")
        moons = []
        for line in f.readlines():
            matches = pattern.match(line)
            moons.append(Moon(tuple(int(m) for m in matches.groups())))
        return cls(frozenset(moons))

    def step(self) -> MoonArray:
        temp = self.update_velocities()
        return temp.update_positions()

    def update_velocities(self) -> MoonArray:
        velocity_dict = self.calc_velocity_deltas()
        return MoonArray(frozenset(
            m.apply_velocity_delta(tuple(velocity_dict[(a, p)] for a, p in zip(["x", "y", "z"], m.position)))
            for m in self.moons
        ))

    def update_positions(self) -> MoonArray:
        return MoonArray(frozenset(m.advance() for m in self.moons))

    def calc_velocity_deltas(self) -> dict[(str, int), int]:
        """Calculate moon velocities

        Returns
        -------
        dict[(str, int), int]
            A dictionary mapping an axis ("x", "y", or "z") and position on that axis to its velocity.
        """

        def count_values_to(positions):
            result = []
            last_pos = None
            pos_count = 0
            for i, p in enumerate(positions):
                if p == last_pos:
                    result.append(pos_count)
                else:
                    pos_count = i
                    result.append(pos_count)
                last_pos = p
            return result

        def values_to_dict(positions, axis):
            positions = sorted(positions)
            positions_less_than = count_values_to(positions)
            positions_greater_than = list(reversed(count_values_to(reversed(positions))))
            return {
                (axis, p): gt - lt
                for p, gt, lt in zip(positions, positions_greater_than, positions_less_than)
            }

        xs, ys, zs = zip(*(m.position for m in self.moons))
        return values_to_dict(xs, "x") | values_to_dict(ys, "y") | values_to_dict(zs, "z")

    def total_energy(self) -> int:
        return sum(m.total_energy() for m in self.moons)


SAMPLE_INPUT = """\
<x=-1, y=0, z=2>
<x=2, y=-10, z=-7>
<x=4, y=-8, z=8>
<x=3, y=5, z=-1>
"""


@pytest.fixture
def sample_array():
    with StringIO(SAMPLE_INPUT) as f:
        return MoonArray.read_array(f)


def test_create_moon_array(sample_array):
    assert len(sample_array.moons) == 4


@pytest.mark.parametrize(
    ("step_count", "expected"),
    [
        (
            0,
            {
                ("x", -1): 3,
                ("x", 2): 1,
                ("x", 3): -1,
                ("x", 4): -3,
                ("y", -10): 3,
                ("y", -8): 1,
                ("y", 0): -1,
                ("y", 5): -3,
                ("z", -7): 3,
                ("z", -1): 1,
                ("z", 2): -1,
                ("z", 8): -3,
            },
        ),
        (
            1,
            {
                ("x", 1): 3,
                ("x", 2): 0,
                ("x", 3): -3,
                ("y", -7): 2,
                ("y", -1): -1,
                ("y", 2): -3,
                ("z", -4): 3,
                ("z", 0): 1,
                ("z", 1): -1,
                ("z", 5): -3,
            }
        )
    ],
)
def test_calc_velocity_delta(sample_array, step_count, expected):
    array = sample_array
    for _ in range(step_count):
        array = array.step()
    assert array.calc_velocity_deltas() == expected


@pytest.mark.parametrize(
    ("step_count", "expected"),
    [
        (
            1,
            MoonArray(
                frozenset([
                    Moon((2, -1, 1), (3, -1, -1)),
                    Moon((3, -7, -4), (1, 3, 3)),
                    Moon((1, -7, 5), (-3, 1, -3)),
                    Moon((2, 2, 0), (-1, -3, 1)),
                ])
            ),
        ),
        (
            2,
            MoonArray(
                frozenset([
                    Moon((5, -3, -1), (3, -2, -2)),
                    Moon((1, -2, 2), (-2, 5, 6)),
                    Moon((1, -4, -1), (0, 3, -6)),
                    Moon((1, -4, 2), (-1, -6, 2)),
                ])
            ),
        ),
        (
            10,
            MoonArray(
                frozenset([
                    Moon((2, 1, -3), (-3, -2, 1)),
                    Moon((1, -8, 0), (-1, 1, 3)),
                    Moon((3, -6, 1), (3, 2, -3)),
                    Moon((2, 0, 4), (1, -1, -1)),
                ])
            )
        )
    ],
)
def test_one_step(sample_array, step_count, expected):
    array = sample_array
    for _ in range(step_count):
        array = array.step()
    assert array == expected


def test_energy(sample_array):
    array = sample_array
    for _ in range(10):
        array = array.step()
    assert array.total_energy() == 179
