"""Advent of Code 2019, day 23: https://adventofcode.com/2019/day/23"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(23, 2019)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        router = Router(program, 50)
        return router.run()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            program = IntcodeMachine.read_buffer(f)
        router = Router(program, 50)
        return router.run(halt_on_nat=False)


class NIC:
    def __init__(self, program: list[int], queue: PacketQueue):
        self.controller = IntcodeMachine(program, input_stream=queue)


class PacketQueue(Iterator[int]):
    def __init__(self, address: int):
        self.address = address
        self.queue: deque[int] = deque()
        self.queue.append(address)
        self.was_read = False

    def __next__(self):
        self.was_read = True
        if self.queue:
            value = self.queue.popleft()
            log.debug("Queue %d sending %d.", self.address, value)
            return value
        return -1

    def send(self, value: int):
        self.queue.append(value)


@dataclass
class Receiver:
    contents: deque[int] = field(default_factory=deque)


class Router:
    def __init__(self, program: list[int], node_count: int):
        self.node_count = node_count
        self.queues = [PacketQueue(n) for n in range(node_count)]
        self.nodes = [NIC(program.copy(), queue) for queue in self.queues]
        self.receivers = [deque() for _ in range(node_count)]

    def run(self, halt_on_nat: bool = True) -> int:
        nat = (0, 0)
        last_nat_y: Optional[int] = None
        while True:
            is_idle: bool = all(len(q.queue) == 0 for q in self.queues)
            for i, node in enumerate(self.nodes):
                self.queues[i].was_read = False
                response = node.controller.multi_step()
                while response is not None:
                    is_idle = False
                    log.debug("Received response %d from %d.", response, i)
                    rec = self.receivers[i]
                    rec.append(response)
                    if len(rec) > 2:
                        address = rec.popleft()
                        x = rec.popleft()
                        y = rec.popleft()
                        if address == 255:
                            if halt_on_nat:
                                return y
                            nat = x, y
                        else:
                            log.debug("Sending (%d, %d) to %d.", x, y, address)
                            self.queues[address].send(x)
                            self.queues[address].send(y)
                    response = node.controller.multi_step()
            all_tried_to_read = all(q.was_read for q in self.queues)
            if all_tried_to_read and is_idle:
                x, y = nat
                if y == last_nat_y:
                    return y
                self.queues[0].send(x)
                self.queues[0].send(y)
                last_nat_y = y
