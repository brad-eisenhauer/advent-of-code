""" Advent of Code 2021, Day 16: https://adventofcode.com/2021/day/16 """
import operator
from abc import ABC, abstractmethod
from functools import reduce
from itertools import islice
from typing import Iterator, Sequence

import pytest

from advent_of_code.base import Solution

bit = int
OPERATORS = {
    0: operator.add,
    1: operator.mul,
    2: min,
    3: max,
    5: operator.gt,
    6: operator.lt,
    7: operator.eq,
}


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2021, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            hex = f.readline()
        bits = generate_bits(hex)
        result, _ = evaluate_packet(bits)
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            hex = f.readline()
        bits = generate_bits(hex)
        _, result = evaluate_packet(bits)
        return result


# region AST solution


class Packet(ABC):
    def __init__(self, packet_type: int, version: int):
        self.type = packet_type
        self.version = version

    @abstractmethod
    def sum_of_versions(self):
        ...

    @abstractmethod
    def evaluate(self) -> int:
        ...


class Literal(Packet):
    def __init__(self, packet_type: int, version: int, value: int):
        super().__init__(packet_type, version)
        self.value = value

    def sum_of_versions(self):
        return self.version

    def evaluate(self) -> int:
        return self.value


class Operation(Packet):
    BINARY_OPERATIONS = {5, 6, 7}

    def __init__(self, packet_type: int, version: int, operands: Sequence[Packet]):
        if packet_type in self.BINARY_OPERATIONS and len(operands) != 2:
            raise ValueError(
                f"Binary operation ({packet_type}) requires exactly 2 operands; "
                f"received {len(operands)}."
            )
        super().__init__(packet_type, version)
        self.operands = operands

    def sum_of_versions(self):
        return self.version + sum(p.sum_of_versions() for p in self.operands)

    def evaluate(self) -> int:
        op = OPERATORS[self.type]
        return int(reduce(op, (p.evaluate() for p in self.operands)))


def parse_packet(bits: Iterator[bit]) -> Packet:
    version = value_of_bits(bits, 3)
    packet_type = value_of_bits(bits, 3)

    if packet_type == 4:
        value = read_literal_value(bits)
        return Literal(packet_type, version, value)

    # Operator packet
    if value_of_bits(bits, 1):
        packet_count = value_of_bits(bits, 11)
        sub_packets = [parse_packet(bits) for _ in range(packet_count)]
    else:
        sub_packet_length = value_of_bits(bits, 15)
        sub_packet_bits = islice(bits, sub_packet_length)
        sub_packets = list(parse_all_packets(sub_packet_bits))
    return Operation(packet_type, version, sub_packets)


def parse_all_packets(bits: Iterator[bit]) -> Iterator[Packet]:
    try:
        while packet := parse_packet(bits):
            yield packet
    except StopIteration:
        ...


# endregion


# region Immediate evaluation


def evaluate_packet(bits: Iterator[bit]) -> tuple[int, int]:
    version = value_of_bits(bits, 3)
    packet_type = value_of_bits(bits, 3)

    if packet_type == 4:
        return version, read_literal_value(bits)

    # Operator packets
    if value_of_bits(bits, 1):
        packet_count = value_of_bits(bits, 11)
        sub_packet_values = (evaluate_packet(bits) for _ in range(packet_count))
    else:
        sub_packet_length = value_of_bits(bits, 15)
        sub_packet_bits = islice(bits, sub_packet_length)
        sub_packet_values = evaluate_all_packets(sub_packet_bits)
    versions, values = zip(*sub_packet_values)
    return sum(versions) + version, reduce(OPERATORS[packet_type], values)


def evaluate_all_packets(bits: Iterator[bit]) -> Iterator[int]:
    try:
        while True:
            yield evaluate_packet(bits)
    except StopIteration:
        ...


# endregion


def generate_bits(hex: str) -> Iterator[bit]:
    for c in hex:
        value = int(c, 16)
        for offset in range(3, -1, -1):
            yield 1 & (value >> offset)


def value_of_bits(bits: Iterator[bit], length: int) -> int:
    result = 0
    for _ in range(length):
        result <<= 1
        result += next(bits)
    return result


def read_literal_value(bits: Iterator[bit]) -> int:
    value = 0
    keep_reading = True
    while keep_reading:
        keep_reading = value_of_bits(bits, 1)
        value <<= 4
        value += value_of_bits(bits, 4)
    return value


# region Tests


def sum_of_versions_ast(bits: Iterator[bit]) -> int:
    packet = parse_packet(bits)
    return packet.sum_of_versions()


def sum_of_versions_immediate(bits: Iterator[bit]) -> int:
    result, _ = evaluate_packet(bits)
    return result


def evaluate_ast(bits: Iterator[bit]) -> int:
    packet = parse_packet(bits)
    return packet.evaluate()


def evaluate_immediate(bits: Iterator[bit]) -> int:
    _, result = evaluate_packet(bits)
    return result


@pytest.mark.parametrize(
    ("hex", "expected", "method"),
    (
        params
        for method in (sum_of_versions_ast, sum_of_versions_immediate)
        for params in (
            ("8A004A801A8002F478", 16, method),
            ("620080001611562C8802118E34", 12, method),
            ("C0015000016115A2E0802F182340", 23, method),
            ("A0016C880162017C3686B18A3D4780", 31, method),
        )
    ),
)
def test_sum_of_versions(hex, expected, method):
    bits = generate_bits(hex)
    assert method(bits) == expected
    assert sum(bits) == 0  # Have we consumed all the meaningful bits?


@pytest.mark.parametrize(
    ("hex", "expected", "method"),
    (
        params
        for method in (evaluate_ast, evaluate_immediate)
        for params in (
            ("C200B40A82", 3, method),
            ("04005AC33890", 54, method),
            ("880086C3E88112", 7, method),
            ("CE00C43D881120", 9, method),
            ("D8005AC2A8F0", 1, method),
            ("F600BC2D8F", 0, method),
            ("9C005AC2F8F0", 0, method),
            ("9C0141080250320F1802104A08", 1, method),
            ("D24A", 42, method),  # literal 42 (multiple quads)
            ("A600AC3587", 42, method),  # product 6 * 7
        )
    ),
)
def test_evaluate(hex, expected, method):
    bits = generate_bits(hex)
    assert method(bits) == expected
    assert sum(bits) == 0  # Have we consumed all the meaningful bits?


def test_generate_bits():
    assert list(generate_bits("50A")) == [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]


def test_value_of_bits():
    bits = iter((1, 0, 1, 0, 0, 1, 0, 0))
    assert value_of_bits(bits, 6) == 41
    assert sum(1 for b in bits if b == 0) == 2  # 2 zeros left over


# endregion
