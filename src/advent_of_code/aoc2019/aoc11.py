from typing import Iterator

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution


class AocSolution(Solution):
    def __init__(self):
        super().__init__(11, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        robot = Robot(program)
        robot.run()
        return len(robot.panels)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        robot = Robot(program)
        robot.panels[(0, 0)] = 1
        robot.run()
        robot.print_panels()


class Robot:
    BLOCK = "▒"

    def __init__(self, program):
        self.brain = IntcodeMachine(program, input_stream=self.eyes())
        self.panels: dict[tuple[int, int], int] = {}
        self.location = (0, 0)
        self.orientation = (0, -1)
        self.input_stream = self.brain.run()

    def eyes(self) -> Iterator[int]:
        while True:
            yield self.panels.get(self.location, 0)

    def run(self):
        try:
            while True:
                color = next(self.input_stream)
                direction = next(self.input_stream)
                self.panels[self.location] = color
                self.turn(direction)
                self.advance()
        except StopIteration:
            ...

    def turn(self, direction: int):
        x, y = self.orientation
        match direction:
            case 0:  # turn left
                self.orientation = y, -x
            case 1:  # turn right
                self.orientation = -y, x
            case _:
                raise ValueError(f"Unrecognized turn instruction: {direction}")

    def advance(self):
        self.location = tuple(a + b for a, b in zip(self.location, self.orientation))

    def print_panels(self):
        mins = maxes = (0, 0)
        for panel in self.panels:
            mins = tuple(min(*xs) for xs in zip(mins, panel))
            maxes = tuple(max(*xs) for xs in zip(maxes, panel))
        for row_idx in range(mins[1], maxes[1] + 1):
            row = "".join(
                self.BLOCK if self.panels.get((col_idx, row_idx)) else " "
                for col_idx in range(mins[0], maxes[0] + 1)
            )
            print(row)