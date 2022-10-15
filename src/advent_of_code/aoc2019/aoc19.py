"""Advent of Code 2019, day 19: https://adventofcode.com/2019/day/19"""
import logging
import math
from collections import deque

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution

Point = tuple[int, ...]

log = logging.getLogger("aoc")


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(19, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        ds = DroneSystem(program)
        return ds.count_covered_area(50)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        ds = DroneSystem(program)
        x, y = ds.find_nearest_covered_window(100)
        return 10_000 * x + y


class DroneSystem:
    def __init__(self, program: list[int]):
        self.program = program

    def coords_in_beam(self, coords: Point) -> bool:
        controller = IntcodeMachine(self.program.copy(), iter(coords))
        [result] = controller.run()
        return result

    def find_nearest_covered_window(self, size: int) -> Point:
        min_outside_upper_tan, max_outside_lower_tan, _, _ = self.estimate_bounds(2 * size)

        def calc_x():
            return math.ceil(
                (size - 1)
                * (max_outside_lower_tan + 1)
                / (min_outside_upper_tan - max_outside_lower_tan)
            )

        def calc_y(x):
            return math.ceil((x + size - 1) * max_outside_lower_tan)

        x = calc_x()
        y = calc_y(x)
        while True:
            fits = True
            next_x, next_y = x, y
            if not self.coords_in_beam((x + size - 1, y)):
                max_outside_lower_tan = max(max_outside_lower_tan, y / (x + size - 1))
                next_x = calc_x()
                next_y = max(y + 1, calc_y(next_x))
                fits = False

            if not self.coords_in_beam((x, y + size - 1)):
                min_outside_upper_tan = min(min_outside_upper_tan, (y + size - 1) / x)
                next_x = max(next_x, x + 1, calc_x())
                next_y = max(next_y, calc_y(next_x))
                fits = False

            if fits:
                return x, y

            x, y = next_x, next_y

    def estimate_bounds(self, dim: int) -> tuple[float, float, float, float]:
        """Estimate boundary angles of the tractor beam.

        Returned values are tangents of known bounds of the beam area.

        Parameters
        ----------
        dim
            Size of bounding box to use to create estimates. Larger values yield more precise
            estimates, but take longer to compute.

        Returns
        -------
        tuple[float, float, float, float]
            Respectively:
            - Tangent of minimum known angle outside the upper bound of the beam
            - Tangent of maximum known angle outside the lower bound of the beam
            - Tangent of maximum known angle inside the upper bound of the beam
            - Tangent of minimum known angle inside the lower bound of the beam
        """
        def find_middle(left: Point, right: Point) -> Point:
            return tuple((a + b) // 2 for a, b in zip(left, right))

        windows: deque[tuple[Point, Point, Point]] = deque()
        windows.append(((dim, dim), (dim, 0), (0, dim)))
        while windows:
            middle, lower, upper = windows.popleft()
            if self.coords_in_beam(middle):
                max_known_outside_lower = lower
                min_known_outside_upper = upper
                min_known_inside_lower = max_known_inside_upper = middle
                break
            else:
                windows.append((find_middle(middle, upper), middle, upper))
                windows.append((find_middle(middle, lower), lower, middle))

        while max(abs(a - b) for a, b in zip(max_known_inside_upper, min_known_outside_upper)) > 1:
            middle = find_middle(max_known_inside_upper, min_known_outside_upper)
            if self.coords_in_beam(middle):
                max_known_inside_upper = middle
            else:
                min_known_outside_upper = middle

        while max(abs(a - b) for a, b in zip(min_known_inside_lower, max_known_outside_lower)) > 1:
            middle = find_middle(min_known_inside_lower, max_known_outside_lower)
            if self.coords_in_beam(middle):
                min_known_inside_lower = middle
            else:
                max_known_outside_lower = middle

        return (
            min_known_outside_upper[1] / min_known_outside_upper[0],
            max_known_outside_lower[1] / max_known_outside_lower[0],
            max_known_inside_upper[1] / max_known_inside_upper[0],
            min_known_inside_lower[1] / min_known_inside_lower[0],
        )

    def count_covered_area(self, dim: int) -> int:
        (
            min_outside_upper_tan,
            max_outside_lower_tan,
            max_inside_upper_tan,
            min_inside_lower_tan,
        ) = self.estimate_bounds(dim)

        result = 1
        for x in range(1, dim):
            max_known_outside_lower = math.floor(min(dim - 1, x * max_outside_lower_tan))
            min_known_inside_lower = math.ceil(min(dim, x * min_inside_lower_tan))
            max_known_inside_upper = math.floor(min(dim - 1, x * max_inside_upper_tan))
            min_known_outside_upper = math.ceil(min(dim, x * min_outside_upper_tan))

            while True:
                if max_known_outside_lower + 1 == min_known_inside_lower:
                    break
                y = (max_known_outside_lower + min_known_inside_lower) // 2
                if self.coords_in_beam((x, y)):
                    log.debug("Drone affected by tractor beam.")
                    min_known_inside_lower = min(min_known_inside_lower, y)
                    max_known_inside_upper = max(max_known_inside_upper, y)
                else:
                    log.debug("Drone not affected by tractor beam.")
                    max_known_outside_lower = max(max_known_outside_lower, y)

            while True:
                if max_known_inside_upper + 1 == min_known_outside_upper:
                    break
                y = (max_known_inside_upper + min_known_outside_upper) // 2
                if self.coords_in_beam((x, y)):
                    log.debug("Drone affected by tractor beam.")
                    min_known_inside_lower = min(min_known_inside_lower, y)
                    max_known_inside_upper = max(max_known_inside_upper, y)
                else:
                    log.debug("Drone not affected by tractor beam.")
                    min_known_outside_upper = min(min_known_outside_upper, y)

            result += min_known_outside_upper - min_known_inside_lower

            min_inside_lower_tan = min(min_inside_lower_tan, min_known_inside_lower / x)
            max_outside_lower_tan = max(max_outside_lower_tan, max_known_outside_lower / x)
            max_inside_upper_tan = max(max_inside_upper_tan, max_known_inside_upper / x)
            min_outside_upper_tan = min(min_outside_upper_tan, min_known_outside_upper / x)

        return result
