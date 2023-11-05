"""Advent of Code 2019, day 23: https://adventofcode.com/2019/day/23"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

from advent_of_code.aoc2019.intcode import IntcodeMachine
from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(23, 2019, **kwargs)

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
class Packet:
    destination: int
    content: tuple[int, int]


class Halt(Exception):
    def __init__(self, value: int):
        self.value = value


class Router:
    def __init__(self, program: list[int], node_count: int):
        self.node_count = node_count
        self.queues = [PacketQueue(n) for n in range(node_count)]
        self.nics = [IntcodeMachine(program.copy(), queue) for queue in self.queues]

    def run(self, halt_on_nat: bool = True) -> int:
        nat = (0, 0)
        last_nat_y: Optional[int] = None
        try:
            while True:
                is_idle, nat = self._run_nics(halt_on_nat, nat)
                all_tried_to_read = all(q.was_read for q in self.queues)
                if all_tried_to_read and is_idle:
                    last_nat_y = self._handle_idle_state(last_nat_y, nat)
        except Halt as h:
            return h.value

    def _handle_idle_state(self, last_nat_y, nat):
        x, y = nat
        if y == last_nat_y:
            raise Halt(y)
        self.queues[0].send(x)
        self.queues[0].send(y)
        return y

    def _run_nics(self, halt_on_nat, nat):
        is_idle: bool = all(len(q.queue) == 0 for q in self.queues)
        for i, nic in enumerate(self.nics):
            for packet in self._run_single_nic(nic):
                log.debug("Received packet %s from %d.", packet, i)
                is_idle = False
                nat = self._handle_packet(halt_on_nat, nat, packet)
        return is_idle, nat

    def _handle_packet(self, halt_on_nat, nat, packet):
        x, y = packet.content
        if packet.destination == 255:
            if halt_on_nat:
                raise Halt(y)
            nat = packet.content
        else:
            log.debug("Sending %s to %d.", packet.content, packet.destination)
            self.queues[packet.destination].send(x)
            self.queues[packet.destination].send(y)
        return nat

    def _run_single_nic(self, nic: IntcodeMachine) -> Iterator[Packet]:
        nic.input_stream.was_read = False
        response = nic.multi_step()
        while response is not None:
            address = response
            x = nic.multi_step()
            y = nic.multi_step()
            yield Packet(address, (x, y))
            response = nic.multi_step()
