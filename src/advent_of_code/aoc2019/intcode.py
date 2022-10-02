from dataclasses import dataclass, field
from typing import Optional, TextIO, Iterator, TypeVar

T = TypeVar("T")


class IllegalOperationError(RuntimeError):
    ...


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
            return self.process_opcode(op_code, modes)
        except TypeError as e:
            raise IllegalOperationError("Attempted to run a halted machine.") from e

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
            case 1:  # add
                left, right = self.read_values(modes, 2)
                result_idx = self.read(self.pointer + 3)
                self.write(result_idx, left + right)
                self.pointer += 4
            case 2:  # multiply
                left, right = self.read_values(modes, 2)
                result_idx = self.read(self.pointer + 3)
                self.write(result_idx, left * right)
                self.pointer += 4
            case 3:  # input
                address = self.read(self.pointer + 1)
                self.write(address, next(self.input_stream))
                self.pointer += 2
            case 4:  # output
                [output] = self.read_values(modes, 1)
                self.pointer += 2
            case 5:  # jump-if-true
                condition, address = self.read_values(modes, 2)
                if condition:
                    self.pointer = address
                else:
                    self.pointer += 3
            case 6:  # jump-if-false
                condition, address = self.read_values(modes, 2)
                if not condition:
                    self.pointer = address
                else:
                    self.pointer += 3
            case 7:  # less than
                left, right = self.read_values(modes, 2)
                address = self.read(self.pointer + 3)
                self.write(address, int(left < right))
                self.pointer += 4
            case 8:  # equals
                left, right = self.read_values(modes, 2)
                address = self.read(self.pointer + 3)
                self.write(address, int(left == right))
                self.pointer += 4
            case 9:  # relative base adjustment
                [adjustment] = self.read_values(modes, 1)
                self.relative_base += adjustment
                self.pointer += 2
            case 99:  # halt
                self.pointer = None
            case _:
                raise ValueError(f"Unrecognized op code: {op_code} at index {self.pointer}")
        return output

    def read_values(self, modes: int, length: int) -> Iterator[int]:
        for offset in range(1, length + 1):
            mode = modes % 10
            modes //= 10
            match mode:
                case 0:  # position mode
                    index = self.read(self.pointer + offset)
                    yield self.read(index)
                case 1:  # immediate mode
                    yield self.read(self.pointer + offset)
                case 2:  # relative mode
                    relative = self.read(self.pointer + offset)
                    yield self.read(self.offset_base + relative)
                case _:
                    raise ValueError(f"Unrecognized parameter mode: {mode}")
