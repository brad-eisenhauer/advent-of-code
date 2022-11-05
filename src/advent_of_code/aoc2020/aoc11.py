"""Advent of Code 2020, day 11: https://adventofcode.com/2020/day/11"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from io import StringIO
from typing import Optional, TextIO

import pytest

from advent_of_code.base import Solution


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(11, 2020, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            initial_state = Seats.read(f)
        last_state = initial_state
        while (state := last_state.next_state()) != last_state:
            last_state = state
        return last_state.count_occupied_seats()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            initial_state = Seats.read(f)
        initial_state = replace(initial_state, line_of_sight=True, crowd_threshold=5)
        last_state = initial_state
        while (state := last_state.next_state()) != last_state:
            last_state = state
        return last_state.count_occupied_seats()


class Seat:
    @abstractmethod
    def occupied(self) -> bool:
        ...

    @classmethod
    def from_char(cls, c: str) -> Optional[Seat]:
        match c:
            case "#":
                return OccupiedSeat()
            case "L":
                return EmptySeat()
            case _:
                return None

    def __bool__(self) -> bool:
        return self.occupied()


class OccupiedSeat(Seat):
    _instance: Optional[OccupiedSeat] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(cls, OccupiedSeat).__new__(cls)
        return cls._instance

    def occupied(self) -> bool:
        return True

    def __str__(self):
        return "#"


class EmptySeat(Seat):
    _instance: Optional[EmptySeat] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(cls, EmptySeat).__new__(cls)
        return cls._instance

    def occupied(self) -> bool:
        return False

    def __str__(self):
        return "L"


@dataclass(frozen=True)
class Seats:
    layout: tuple[tuple[Optional[Seat], ...], ...]
    line_of_sight: bool = False
    crowd_threshold: int = 4

    @classmethod
    def read(cls, f: TextIO) -> Seats:
        return cls(tuple(tuple(Seat.from_char(c) for c in line) for line in f.readlines()))

    def __str__(self):
        return "\n".join(
            "".join(str(seat) if seat is not None else "." for seat in row) for row in self.layout
        )

    def __repr__(self):
        return str(self)

    def count_occupied_seats(self) -> int:
        return sum(1 for row in self.layout for seat in row if seat)

    def next_state(self) -> Seats:
        neighbor_counts = [[0] * len(row) for row in self.layout]
        for y, row in enumerate(self.layout):
            for x, seat in enumerate(row):
                if not seat:
                    continue
                for dx, dy in [
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (-1, 1),
                    (-1, 0),
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                ]:
                    try:
                        for n in range(1, len(self.layout) if self.line_of_sight else 2):
                            neighbor_x = x + n * dx
                            neighbor_y = y + n * dy
                            if neighbor_x < 0 or neighbor_y < 0:
                                raise IndexError()
                            if self.layout[neighbor_y][neighbor_x] is not None:
                                neighbor_counts[neighbor_y][neighbor_x] += 1
                                raise StopIteration()
                    except (IndexError, StopIteration):
                        pass

        def occupy_seat(seat: Optional[Seat], neighbor_count: int) -> Optional[Seat]:
            if seat is None:
                return None
            match seat.occupied(), neighbor_count:
                case False, 0:
                    return OccupiedSeat()
                case True, n if n >= self.crowd_threshold:
                    return EmptySeat()
                case _:
                    return seat

        return replace(
            self,
            layout=tuple(
                tuple(occupy_seat(seat, neighbor_counts[y][x]) for x, seat in enumerate(row))
                for y, row in enumerate(self.layout)
            ),
        )


SAMPLE_INPUT = [
    """\
L.LL.LL.LL
LLLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLLL
L.LLLLLL.L
L.LLLLL.LL
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT[0]) as f:
        yield f


@pytest.mark.parametrize("step_count", list(range(1, 6)))
def test_next_state_p1(sample_input, step_count):
    successor_states = [
        """\
#.##.##.##
#######.##
#.#.#..#..
####.##.##
#.##.##.##
#.#####.##
..#.#.....
##########
#.######.#
#.#####.##
""",
        """\
#.LL.L#.##
#LLLLLL.L#
L.L.L..L..
#LLL.LL.L#
#.LL.LL.LL
#.LLLL#.##
..L.L.....
#LLLLLLLL#
#.LLLLLL.L
#.#LLLL.##
""",
        """\
#.##.L#.##
#L###LL.L#
L.#.#..#..
#L##.##.L#
#.##.LL.LL
#.###L#.##
..#.#.....
#L######L#
#.LL###L.L
#.#L###.##
""",
        """\
#.#L.L#.##
#LLL#LL.L#
L.L.L..#..
#LLL.##.L#
#.LL.LL.LL
#.LL#L#.##
..L.L.....
#L#LLLL#L#
#.LLLLLL.L
#.#L#L#.##
""",
        """\
#.#L.L#.##
#LLL#LL.L#
L.#.L..#..
#L##.##.L#
#.#L.LL.LL
#.#L#L#.##
..L.L.....
#L#L##L#L#
#.LLLLLL.L
#.#L#L#.##
""",
    ]

    with StringIO(successor_states[step_count - 1]) as f:
        expected = Seats.read(f)
    state = Seats.read(sample_input)
    for _ in range(step_count):
        state = state.next_state()
    assert state == expected


@pytest.mark.parametrize("step_count", list(range(1, 7)))
def test_next_state_p2(sample_input, step_count):
    successor_states = [
        """\
#.##.##.##
#######.##
#.#.#..#..
####.##.##
#.##.##.##
#.#####.##
..#.#.....
##########
#.######.#
#.#####.##
""",
        """\
#.LL.LL.L#
#LLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLL#
#.LLLLLL.L
#.LLLLL.L#
""",
        """\
#.L#.##.L#
#L#####.LL
L.#.#..#..
##L#.##.##
#.##.#L.##
#.#####.#L
..#.#.....
LLL####LL#
#.L#####.L
#.L####.L#
""",
        """\
#.L#.L#.L#
#LLLLLL.LL
L.L.L..#..
##LL.LL.L#
L.LL.LL.L#
#.LLLLL.LL
..L.L.....
LLLLLLLLL#
#.LLLLL#.L
#.L#LL#.L#
""",
        """\
#.L#.L#.L#
#LLLLLL.LL
L.L.L..#..
##L#.#L.L#
L.L#.#L.L#
#.L####.LL
..#.#.....
LLL###LLL#
#.LLLLL#.L
#.L#LL#.L#
""",
        """\
#.L#.L#.L#
#LLLLLL.LL
L.L.L..#..
##L#.#L.L#
L.L#.LL.L#
#.LLLL#.LL
..#.L.....
LLL###LLL#
#.LLLLL#.L
#.L#LL#.L#
""",
    ]
    with StringIO(successor_states[step_count - 1]) as f:
        expected = Seats.read(f)
    expected = replace(expected, line_of_sight=True, crowd_threshold=5)
    state = Seats.read(sample_input)
    state = replace(state, line_of_sight=True, crowd_threshold=5)

    for _ in range(step_count):
        state = state.next_state()

    assert state == expected
