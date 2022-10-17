import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional, TextIO, TypeVar

from advent_of_code.errors import IllegalOperationError

T = TypeVar("T")

log = logging.getLogger("aoc")


@dataclass
class IntcodeMachine:
    buffer: list[int]
    input_stream: Iterator[int] = field(default_factory=lambda: iter(()))
    pointer: Optional[int] = 0
    relative_base: int = 0

    @staticmethod
    def read_buffer(f: TextIO) -> list[int]:
        return [int(n) for n in f.readline().split(",")]

    def run(self) -> Iterator[int]:
        while self.pointer is not None:
            output = self.step()
            if output is not None:
                yield output

    def step(self) -> Optional[int]:
        try:
            op_code = self.buffer[self.pointer] % 100
            modes = self.buffer[self.pointer] // 100
        except TypeError as e:
            raise IllegalOperationError("Attempted to run a halted machine.") from e
        return self.process_opcode(op_code, modes)

    def multi_step(self, max_steps: int = 50) -> Optional[int]:
        try:
            for _ in range(max_steps):
                result = self.step()
                if result is not None:
                    return result
        except IllegalOperationError:
            ...
        return None

    def read(self, index: int) -> int:
        while index >= len(self.buffer):
            self.buffer.extend([0] * len(self.buffer))
        return self.buffer[index]

    def write(self, index: int, value: int):
        while index >= len(self.buffer):
            self.buffer.extend([0] * len(self.buffer))
        self.buffer[index] = value

    def process_opcode(self, op_code: int, modes: int) -> Optional[int]:
        output: Optional[int] = None
        match op_code:
            case 1:  # add (day 2)
                left, right = self.read_values(modes, 2)
                _, _, result_idx = self.read_address(modes, 3)
                self.write(result_idx, left + right)
                self.pointer += 4
            case 2:  # multiply (day 2)
                left, right = self.read_values(modes, 2)
                _, _, result_idx = self.read_address(modes, 3)
                self.write(result_idx, left * right)
                self.pointer += 4
            case 3:  # input (day 5)
                [address] = self.read_address(modes, 1)
                value = next(self.input_stream)
                self.write(address, value)
                if value != -1:
                    log.debug("Read %d and stored at address %d.", value, address)
                self.pointer += 2
            case 4:  # output (day 5)
                [output] = self.read_values(modes, 1)
                self.pointer += 2
            case 5:  # jump-if-true (day 5)
                condition, address = self.read_values(modes, 2)
                if condition:
                    self.pointer = address
                else:
                    self.pointer += 3
            case 6:  # jump-if-false (day 5)
                condition, address = self.read_values(modes, 2)
                if not condition:
                    self.pointer = address
                else:
                    self.pointer += 3
            case 7:  # less than (day 5)
                left, right = self.read_values(modes, 2)
                _, _, address = self.read_address(modes, 3)
                self.write(address, int(left < right))
                self.pointer += 4
            case 8:  # equals (day 5)
                left, right = self.read_values(modes, 2)
                _, _, address = self.read_address(modes, 3)
                self.write(address, int(left == right))
                self.pointer += 4
            case 9:  # relative base adjustment (day 9)
                adjustment = next(self.read_values(modes, 1))
                self.relative_base += adjustment
                self.pointer += 2
            case 99:  # halt (day 2)
                self.pointer = None
            case _:
                raise ValueError(f"Unrecognized op code: {op_code} at index {self.pointer}")
        return output

    def read_address(self, modes: int, length: int) -> Iterator:
        for offset in range(1, length + 1):
            mode = modes % 10
            modes //= 10
            match mode:
                case 0:  # position mode (day 5)
                    index = self.read(self.pointer + offset)
                    yield index
                case 1:  # immediate mode (day 5)
                    yield self.pointer + offset
                case 2:  # relative mode (day 9)
                    relative_offset = self.read(self.pointer + offset)
                    yield self.relative_base + relative_offset
                case _:
                    raise ValueError(f"Unrecognized parameter mode: {mode}")

    def read_values(self, modes: int, length: int) -> Iterator[int]:
        for address in self.read_address(modes, length):
            yield self.read(address)
