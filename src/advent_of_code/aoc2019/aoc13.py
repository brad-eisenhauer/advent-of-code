"""Advent of Code 2019, Day 13: https://adventofcode.com/2019/day/13"""
import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import ClassVar, Iterator, Optional

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2019, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        game = Game(IntcodeMachine(program))
        game.run()
        return len(game.blocks)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        program[0] = 2
        game = Game(IntcodeMachine(program))
        game.run()
        return game.get_score()


@dataclass
class Game:
    machine: IntcodeMachine
    board: dict[tuple[int, int], int] = field(default_factory=dict, init=False)
    blocks: set[tuple[int, int]] = field(default_factory=set, init=False)
    ball_history: list[tuple[int, int]] = field(default_factory=list, init=False)
    paddle: Optional[tuple[int, int]] = field(default=None, init=False)

    PRINT_CHARS: ClassVar[dict[int, str]] = {
        0: " ",  # empty
        1: "█",  # wall
        2: "▒",  # block
        3: "=",  # paddle
        4: "0",  # ball
    }
    BALL_CHARS: ClassVar[tuple[str, ...]] = (".", "o", "O", "0")

    def __post_init__(self):
        self.machine.input_stream = self.joystick()

    def block_count(self) -> int:
        return len(self.blocks)

    def run(self):
        instructions = self.machine.run()
        try:
            while True:
                x, y, tile = islice(instructions, 3)
                self.board[(x, y)] = tile

                match tile:
                    case 0:
                        if (x, y) in self.blocks:
                            self.blocks.remove((x, y))
                    case 2:
                        self.blocks.add((x, y))
                    case 3:
                        self.paddle = x, y
                    case 4:
                        self.ball_history.append((x, y))
                        self.print_blocks()
                    case _:
                        ...
        except ValueError:
            ...

    def get_score(self) -> int:
        return self.board[(-1, 0)]

    def print_blocks(self):
        if log.level > logging.DEBUG:
            return
        max_col_idx, max_row_idx = (max(ns) for ns in zip(*self.board.keys()))
        for row_idx in range(0, max_row_idx + 1):
            line = [
                self.PRINT_CHARS[self.board.get((col_idx, row_idx), 0)]
                for col_idx in range(0, max_col_idx + 1)
            ]
            for i, (x, y) in enumerate(self.ball_history[-4:]):
                if y == row_idx:
                    line[x] = self.BALL_CHARS[i]
            log.debug("%s" * len(line), *line)
        log.debug("")

    def joystick(self) -> Iterator[int]:
        while True:
            ball_x = self.ball_history[-1][0]
            paddle_x = self.paddle[0]
            if ball_x < paddle_x:
                result = -1
            elif ball_x == paddle_x:
                result = 0
            else:
                result = 1
            yield result
