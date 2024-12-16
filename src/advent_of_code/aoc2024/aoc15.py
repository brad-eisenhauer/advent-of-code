"""Advent of Code 2024, day 15: https://adventofcode.com/2024/day/15"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from io import StringIO
from typing import IO, Optional, TypeAlias

import pytest

from advent_of_code.base import Solution

Vector: TypeAlias = tuple[int, int]


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(15, 2024, **kwargs)

    def solve_part_one(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            warehouse = Warehouse.read(reader)
            warehouse.run(reader)
        return warehouse.calc_gps()

    def solve_part_two(self, input_file: Optional[IO] = None) -> int:
        with input_file or self.open_input() as reader:
            warehouse = Warehouse2.read(reader)
            warehouse.run(reader)
        warehouse.show()
        return warehouse.calc_gps()


DIRECTIONS: dict[str, Vector] = {"^": (-1, 0), ">": (0, 1), "<": (0, -1), "v": (1, 0)}


@dataclass
class Warehouse:
    robot: Vector
    boxes: set[Vector]
    walls: set[Vector]

    @classmethod
    def read(cls, reader: IO) -> Warehouse:
        robot: Vector | None = None
        boxes: set[Vector] = set()
        walls: set[Vector] = set()
        for i, line in enumerate(reader):
            if not line.strip():
                break
            for j, char in enumerate(line):
                match char:
                    case "#":
                        walls.add((i, j))
                    case "O":
                        boxes.add((i, j))
                    case "@":
                        robot = (i, j)
        return cls(robot, boxes, walls)

    def step(self, direction: str) -> None:
        """Attempt to move the robot one square in the specified direction.

        We need to find the first empty space in the specified direction, beginning from the robot.
        Any intervening boxes will be moved one unit. If we encounter a wall before finding empty
        space, no movement occurs.
        """
        unit = DIRECTIONS[direction]
        looking_at = self.robot
        while True:
            looking_at = vector_add(looking_at, unit)
            if looking_at in self.walls:
                # We're done. No movement is possible.
                return
            if looking_at not in self.boxes:
                # Empty space. Shift any boxes and the robot.
                new_robot_pos = vector_add(self.robot, unit)
                if new_robot_pos != looking_at:
                    # There is at least one box to move. Instead of moving all the boxes, we'll just
                    # shift the first box to the end.
                    self.boxes.remove(new_robot_pos)
                    self.boxes.add(looking_at)
                self.robot = new_robot_pos
                return
            # Otherwise, we keep looking.

    def calc_gps(self) -> int:
        return sum(100 * i + j for i, j in self.boxes)

    def run(self, reader: IO) -> None:
        while line := reader.readline().strip():
            for char in line:
                if char in DIRECTIONS:
                    self.step(char)


@dataclass
class Box:
    pos: Vector
    warehouse: Warehouse2
    _id: uuid.Uuid = field(default_factory=uuid.uuid4)

    def __hash__(self) -> int:
        return hash(self._id)

    def can_move(self, direction: Vector) -> bool:
        adj_pos = self.get_adjacent_pos(direction)
        if any(p in self.warehouse.walls for p in adj_pos):
            return False
        adj_boxes = {b for p in adj_pos if (b := self.warehouse.get_box_at(p)) is not None}
        return all(b.can_move(direction) for b in adj_boxes)

    def get_adjacent_pos(self, direction: Vector) -> list[Vector]:
        if direction[0] != 0:
            return [
                vector_add(p, direction) for p in [self.pos, vector_add(self.pos, DIRECTIONS[">"])]
            ]
        if direction == (0, -1):
            return [vector_add(self.pos, direction)]
        return [vector_add(self.pos, (0, 2))]

    def calc_gps(self) -> int:
        i, j = self.pos
        return 100 * i + j


@dataclass
class Warehouse2:
    robot: Vector
    boxes: dict[Vector, Box]
    walls: set[Vector]

    @classmethod
    def read(cls, reader: IO) -> Warehouse2:
        warehouse = Warehouse2((0, 0), {}, set())
        for i, line in enumerate(reader):
            if not (line := line.rstrip()):
                break
            for j, char in enumerate(line):
                pos = (i, 2 * j)
                match char:
                    case "#":
                        warehouse.walls |= {pos, vector_add(pos, (0, 1))}
                    case "O":
                        box = Box(pos, warehouse)
                        warehouse.boxes[pos] = box
                    case "@":
                        warehouse.robot = pos
        return warehouse

    def move_box(self, box: Box, direction: Vector) -> None:
        # Presume box.can_move(direction) is True.
        # Move any adjoining boxes.
        for p in box.get_adjacent_pos(direction):
            if (b := self.get_box_at(p)) is not None:
                self.move_box(b, direction)
        # Move the current box.
        del self.boxes[box.pos]
        box.pos = vector_add(box.pos, direction)
        self.boxes[box.pos] = box

    def get_box_at(self, pos: Vector) -> Box | None:
        if pos in self.boxes:
            return self.boxes[pos]
        if (adj_pos := vector_add(pos, DIRECTIONS["<"])) in self.boxes:
            return self.boxes[adj_pos]
        return None

    def step(self, direction: Vector) -> None:
        new_robot_pos = vector_add(self.robot, direction)
        if new_robot_pos in self.walls:
            return
        if (box := self.get_box_at(new_robot_pos)) is None:
            self.robot = new_robot_pos
        elif box.can_move(direction):
            self.move_box(box, direction)
            self.robot = new_robot_pos

    def run(self, reader: IO) -> None:
        for char in reader.read():
            if char in DIRECTIONS:
                self.step(DIRECTIONS[char])

    def calc_gps(self) -> int:
        return sum(b.calc_gps() for b in self.boxes.values())

    def show(self) -> None:
        max_i, max_j = max(self.walls)
        for i in range(max_i + 1):
            for j in range(max_j + 1):
                pos = (i, j)
                if pos in self.walls:
                    print("#", end="")
                elif pos == self.robot:
                    print("@", end="")
                elif (box := self.get_box_at(pos)) is not None:
                    if box.pos == pos:
                        print("[", end="")
                    else:
                        print("]", end="")
                else:
                    print(".", end="")
            print()


SAMPLE_INPUTS = [
    """\
##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^
""",
    """\
########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

<^^>>>vv<v>>v<<
""",
]


@pytest.fixture
def sample_input(request):
    return StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)])


@pytest.fixture
def solution():
    return AocSolution()


@pytest.mark.parametrize(
    ("sample_input", "expected"), [(0, 10092), (1, 2028)], indirect=["sample_input"]
)
def test_part_one(solution: AocSolution, sample_input: IO, expected: int):
    assert solution.solve_part_one(sample_input) == expected


@pytest.mark.parametrize(("sample_input", "expected"), [(0, 9021)], indirect=["sample_input"])
def test_part_two(solution: AocSolution, sample_input: IO, expected: int):
    assert solution.solve_part_two(sample_input) == expected
