"""Advent of Code 2023, day 10: https://adventofcode.com/2023/day/10"""
from __future__ import annotations

from collections import deque
from io import StringIO
from typing import IO, Iterable, Optional

import pytest
from rich.console import Console

from advent_of_code.base import Solution


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2023, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            node_map, start = read_network(fp)
        node_distances, *_ = trace_loop(node_map, start)
        return (max(node_distances.values()) + 1) // 2

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as fp:
            node_map, start = read_network(fp)
        node_distances, turn_diff, left_side, right_side = trace_loop(node_map, start)
        loop_nodes = set(node_distances.keys())
        known_interior_nodes = left_side if turn_diff > 0 else right_side
        interior: set[Node] = set()
        frontier = deque(known_interior_nodes)
        while frontier:
            node = frontier.popleft()
            if node in interior:
                continue
            interior.add(node)
            for n in (offset_node(node, o) for o in [(1, 0), (-1, 0), (0, 1), (0, -1)]):
                if n not in node_map:
                    raise ValueError("Found exterior node.")
                if n not in loop_nodes:
                    frontier.append(n)
        # visualize(Console(highlight=False), node_map, loop_nodes, known_interior_nodes, interior)
        return len(interior)


Node = tuple[int, int]
Orientation = tuple[int, int]


def read_network(file: IO) -> tuple[dict[Node, str], Node]:
    node_map: dict[Node, str] = {}
    start_node: Optional[Node] = None
    for row, line in enumerate(file):
        for col, char in enumerate(line.strip()):
            node = (row, col)
            node_map[node] = char
            if char == "S":
                start_node = node
    return node_map, start_node


def can_connect(node_map: dict[Node, str], left: Node, right: Node) -> bool:
    def connects_to(node: Node) -> Iterable[Node]:
        row, col = node
        match (c := node_map.get(node, ".")):
            case "S":
                return [(row, col + 1), (row, col - 1), (row + 1, col), (row - 1, col)]
            case ".":
                return []
            case "F":
                return [(row, col + 1), (row + 1, col)]
            case "J":
                return [(row - 1, col), (row, col - 1)]
            case "7":
                return [(row, col - 1), (row + 1, col)]
            case "L":
                return [(row - 1, col), (row, col + 1)]
            case "|":
                return [(row - 1, col), (row + 1, col)]
            case "-":
                return [(row, col - 1), (row, col + 1)]
        raise ValueError(f"Unrecognized symbol: '{c}'")

    return right in connects_to(left) and left in connects_to(right)


def offset_node(n: Node, o: Orientation) -> Node:
    return tuple(a + b for a, b in zip(n, o))


def trace_loop(
    node_map: dict[Node, str], start: Node
) -> tuple[dict[Node, int], int, set[Node], set[Node]]:
    """Trace the pipes from start.

    Returns
    -------
    tuple[dict[Node, int], int, set[Node], set[Node]]
        - Shortest traversal distance from start node to each node in the loop
        - Difference of left turns - right turns (positive == counterclockwise traversal, negative
          == clockwise traversal)
        - Neighboring nodes on the left side of traversal
        - Neighboring nodes of the right side of traversal
    """
    neighbor_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    distances: dict[Node, int] = {start: 0}
    turn_diff = 0
    left_side: set[Node] = set()
    right_side: set[Node] = set()

    def add_left(node: Node, orientation: Orientation):
        ox, oy = orientation
        left = (-oy, ox)
        left_side.add(offset_node(node, left))

    def add_right(node: Node, orientation: Orientation):
        ox, oy = orientation
        right = (oy, -ox)
        right_side.add(offset_node(node, right))

    # Traverse loop, tracking distances, turns, neighbors
    orientation = next(
        d for d in neighbor_vectors if can_connect(node_map, start, offset_node(start, d))
    )
    add_left(start, orientation)
    add_right(start, orientation)
    last_node = start
    next_node = offset_node(last_node, orientation)
    while True:
        if next_node == start:
            break

        distances[next_node] = distances[last_node] + 1
        add_left(next_node, orientation)
        add_right(next_node, orientation)

        ox, oy = orientation
        match node_map.get(next_node, "."):
            case "-" | "|":
                pass
            case "F" | "J":
                orientation = -oy, -ox
            case "L" | "7":
                orientation = oy, ox
        if orientation == (-oy, ox):  # rotate left
            turn_diff += 1
            add_right(next_node, orientation)
        elif orientation == (oy, -ox):  # rotate right
            turn_diff -= 1
            add_left(next_node, orientation)

        last_node, next_node = next_node, offset_node(next_node, orientation)

    loop_nodes = set(distances.keys())
    return distances, turn_diff, left_side - loop_nodes, right_side - loop_nodes


def visualize(console: Console, node_map: dict[Node, str], *sets: set[Node]):
    colors = ["bright_black", "red", "green"]
    default_color = "blue"
    dims = tuple(max(xs) for xs in zip(*node_map.keys()))
    pipes = {"|": "║", "-": "═", "7": "╗", "J": "╝", "L": "╚", "F": "╔", "S": "S"}
    color_map = [
        [
            next((c for c, s in zip(colors, sets) if (row, col) in s), default_color)
            for col in range(dims[1] + 1)
        ]
        for row in range(dims[0] + 1)
    ]
    for row in range(dims[0] + 1):
        print_str = ""
        current_color = None
        for col in range(dims[1] + 1):
            color = color_map[row][col]
            if current_color is None:
                print_str += f"[{color}]"
                current_color = color
            elif color != current_color:
                print_str += f"[/{current_color}][{color}]"
                current_color = color
            print_str += (
                pipes[node_map[(row, col)]] if (row, col) in sets[0] else node_map[(row, col)]
            )
        print_str += f"[/{current_color}]"
        console.print(print_str)


SAMPLE_INPUTS = [
    """\
..F7.
.FJ|.
SJ.L7
|F--J
LJ...
""",
    """\
FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L
""",
    """\
...........
.S-------7.
.|F-----7|.
.||.....||.
.||.....||.
.|L-7.F-J|.
.|..|.|..|.
.L--J.L--J.
...........
""",
    """\
.F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ...
""",
]


@pytest.fixture()
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


@pytest.fixture()
def solution():
    return AocSolution()


def test_part_one(solution: AocSolution, sample_input: IO):
    assert solution.solve_part_one(sample_input) == 8


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 1), (1, 10), (2, 4), (3, 8)], indirect=["sample_input"]
)
def test_part_two(solution: AocSolution, sample_input: IO, expected: int):
    assert solution.solve_part_two(sample_input) == expected


def test_read_network(sample_input):
    result = read_network(sample_input)
    assert result is not None
