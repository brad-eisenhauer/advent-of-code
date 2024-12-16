"""Advent of Code 2024, day 12: https://adventofcode.com/2024/day/12"""

from __future__ import annotations

from collections import defaultdict, deque
from io import StringIO
from typing import IO, Optional

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(12, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            grid = read_grid(reader)
        regions = find_regions(grid)
        return sum(calc_fencing_cost(region) for region in regions.values())

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            grid = read_grid(reader)
        regions = find_regions(grid)
        return sum(calc_fencing_cost(region, bulk_discount=True) for region in regions.values())


PADDING = 1
I = complex(0, 1)
DIRECTIONS = [1, I, -1, -I]


def read_grid(reader: IO) -> list[str]:
    return [line.strip() for line in reader]


def find_regions(grid: list[str]) -> dict[str, set[complex]]:
    result: dict[str, set[complex]] = defaultdict(set)
    for i, line in enumerate(grid):
        for j, char in enumerate(line):
            result[char].add(complex(i, j))
    return result


def calc_fencing_cost(region: set[complex], bulk_discount: bool = False) -> int:
    unfenced_plots = set(region)
    result = 0
    while unfenced_plots:
        frontier: deque[complex] = deque([unfenced_plots.pop()])
        visited: set[complex] = set()
        while frontier:
            p = frontier.popleft()
            visited.add(p)
            for d in DIRECTIONS:
                if (n := p + d) in region and n not in visited:
                    frontier.append(n)
                    visited.add(n)
        area = len(visited)
        if bulk_discount:
            perimeter = count_sides(visited)
        else:
            perimeter = 0
            for p in visited:
                for d in DIRECTIONS:
                    if p + d not in visited:
                        perimeter += 1
        result += area * perimeter
        unfenced_plots -= visited
    return result


def count_sides(region: set[complex]) -> int:
    """Count the number of sides

    Count sides by traversing the perimeter of the region. Each turn required to return to the
    original starting state (location and facing direction) represents the end of a side and the
    beginning of another.

    The supplied region is presumed to consist of a contiguous set of plots.

    Using clockwise winding we always keep a plot to our right and a non-plot to the left. We have
    to turn when:
    - The plot ahead and to the right is not in the region. (turn right)
    - The plots ahead to both the left and right are in the region. (turn left)
    - If neither condition applies, continue straight.

    To find a valid starting state, we need to find a vertex adjoining at least 1 and not more than
    3 plots. Orientation should be that a plot is behind and to the right, but none to the left.

    To account for interior holes, track vertices we've passed through. Make sure we've covered all
    boundary vertices.
    """
    # Find starting vertex and orientation.
    boundary_vertices: set[complex] = set()
    for plot in region:
        for vertex in [plot, plot + 1, plot + I, plot + 1 + I]:
            neighboring_plots = [vertex, vertex - 1, vertex - I, vertex - 1 - I]
            neighbor_count = sum(1 for p in neighboring_plots if p in region)
            if 1 <= neighbor_count <= 3:
                boundary_vertices.add(vertex)

    side_count = 0
    while boundary_vertices:
        vertex = boundary_vertices.pop()
        for facing in DIRECTIONS:
            plot_behind_right = vertex + facing * complex(-0.5, -0.5) - complex(0.5, 0.5)
            plot_behind_left = vertex + facing * complex(-0.5, 0.5) - complex(0.5, 0.5)
            if plot_behind_right in region and plot_behind_left not in region:
                break
        else:
            raise ValueError("Failed to find a suitable starting direction.")
        initial_state = vertex, facing

        # Trace the perimeter of the region
        turn_count = 0

        def _step() -> None:
            nonlocal turn_count, vertex, facing, boundary_vertices
            plot_ahead_right = vertex + facing * complex(0.5, -0.5) - complex(0.5, 0.5)
            plot_ahead_left = vertex + facing * complex(0.5, 0.5) - complex(0.5, 0.5)
            if plot_ahead_right not in region:
                facing /= I
                turn_count += 1
            elif plot_ahead_left in region:
                facing *= I
                turn_count += 1
            vertex += facing
            boundary_vertices -= {vertex}

        _step()
        while (vertex, facing) != initial_state:
            _step()
        side_count += turn_count

    return side_count


SAMPLE_INPUTS = [
    """\
RRRRIICCFF
RRRRIICCCF
VVRRRCCFFF
VVRCCCJFFF
VVVVCJJCFE
VVIVCCJJEE
VVIIICJJEE
MIIIIIJJEE
MIIISIJEEE
MMMISSJEEE
""",
]


@pytest.fixture()
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture()
def solution():
    return AocSolution()


def test_find_regions(sample_input: IO) -> None:
    grid = read_grid(sample_input)
    regions = find_regions(grid)
    # fmt: off
    expected_regions = {
        "R": {
            complex(0, 0), complex(0, 1), complex(0, 2), complex(0, 3),
            complex(1, 0), complex(1, 1), complex(1, 2), complex(1, 3),
            complex(2, 2), complex(2, 3), complex(2, 4),
            complex(3, 2),
        },
        "I": {
            complex(0, 4), complex(0, 5), complex(1, 4), complex(1, 5),
            complex(5, 2), complex(6, 2), complex(6, 3), complex(6, 4),
            complex(7, 1), complex(7, 2), complex(7, 3), complex(7, 4), complex(7, 5),
            complex(8, 1), complex(8, 2), complex(8, 3), complex(8, 5), complex(9, 3),
        },
        "C": {
            complex(0, 6), complex(0, 7), complex(1, 6), complex(1, 7), complex(1, 8),
            complex(2, 5), complex(2, 6), complex(3, 3), complex(3, 4), complex(3, 5),
            complex(4, 4), complex(4, 7), complex(5, 4), complex(5, 5), complex(6, 5),
        },
    }
    # fmt: on
    for label, region in expected_regions.items():
        assert regions[label] == region


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ({complex(0, 0)}, 4),
        ({complex(0, 0), complex(1, 0), complex(0, 1)}, 6),
        ({complex(0, 1), complex(0, 2), complex(0, 3), complex(1, 1), complex(1, 3)}, 8),
        # fmt: off
        # interior hole at (1, 2)
        (
            {
                complex(0, 1),
                complex(0, 2),
                complex(0, 3),
                complex(1, 1),
                complex(1, 3),
                complex(2, 1),
                complex(2, 2),
                complex(2, 3),
            },
            8,
        ),
        # "C" shape with hole at (1, 2) and opening at (0, 3)
        (
            {
                complex(0, 1),
                complex(0, 2),
                complex(1, 1),
                complex(1, 3),
                complex(2, 1),
                complex(2, 2),
                complex(2, 3),
            },
            10,
        ),
        # fmt: on
    ],
)
def test_count_sides(region: set[complex], expected: int) -> None:
    assert count_sides(region) == expected


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 1930


def test_part_two(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_two(sample_input) == 1206
