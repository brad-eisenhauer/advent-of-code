from dataclasses import dataclass, field
from typing import Optional, TextIO, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class IntcodeMachine:
    buffer: list[int]
    input_stream: Iterator[int] = field(default_factory=lambda: iter(()))
    pointer: Optional[int] = 0

    @staticmethod
    def read_buffer(f: TextIO) -> list[int]:
        return [int(n) for n in f.readline().split(",")]

    def run(self) -> Iterator[int]:
        while self.pointer is not None:
            output = self.step()
            if output is not None:
                yield output

    def step(self) -> Optional[int]:
        op_code = self.buffer[self.pointer] % 100
        modes = self.buffer[self.pointer] // 100
        return self.process_opcode(op_code, modes)

    def process_opcode(self, op_code: int, modes: int) -> Optional[int]:
        output: Optional[int] = None
        match op_code:
            case 1:  # add
                left, right = self.read_values(modes, 2)
                result_idx = self.buffer[self.pointer + 3]
                self.buffer[result_idx] = left + right
                self.pointer += 4
            case 2:  # multiply
                left, right = self.read_values(modes, 2)
                result_idx = self.buffer[self.pointer + 3]
                self.buffer[result_idx] = left * right
                self.pointer += 4
            case 3:  # input
                address = self.buffer[self.pointer + 1]
                self.buffer[address] = next(self.input_stream)
                self.pointer += 2
            case 4:  # output
                [output] = self.read_values(modes, 1)
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
                case 0:
                    index = self.buffer[self.pointer + offset]
                    yield self.buffer[index]
                case 1:
                    yield self.buffer[self.pointer + offset]
                case _:
                    raise ValueError(f"Unrecognized parameter mode: {mode}")
