import math
import operator
from collections import Counter
from functools import reduce
from itertools import count, product as cross_product
from typing import Iterable, Iterator, Optional, TypeVar, Union

T = TypeVar("T")


def greatest_common_divisor(a: int, b: int) -> int:
    gcd, _, _ = extended_gcd(a, b)
    return gcd


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return abs(a), 1, 0
    gcd, x, y = extended_gcd(b, a % b)
    return gcd, y, x - (a // b) * y


def mod_inverse(a: int, b: int) -> int:
    g, x, _ = extended_gcd(a, b)
    if g != 1:
        raise ValueError(f"{a} and {b} are not coprime; cannot calculate modular inverse.")
    return x


def mod_power(base: int, exp: int, mod: int) -> int:
    if mod == 1:
        return 0
    result = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


def least_common_multiple(a: int, b: int) -> int:
    return a // greatest_common_divisor(a, b) * b


def clamp(n: int, mn: int, mx: int) -> int:
    return min(mx, max(mn, n))


def sign(n: int) -> int:
    if n == 0:
        return 0
    return n // abs(n)


def product(xs: Iterable[T]) -> Union[T, int]:
    return reduce(operator.mul, xs, 1)


def intersect_ranges(left: range, right: range) -> range:
    inter_start = max(left.start, right.start)
    inter_stop = min(left.stop, right.stop)
    return range(inter_start, inter_stop)


def union_ranges(left: range, right: range) -> Optional[range]:
    x_range = intersect_ranges(left, right)
    if not (x_range):
        return None
    union_start = min(left.start, right.start)
    union_stop = max(left.stop, right.stop)
    return range(union_start, union_stop)


def pseudoprimes() -> Iterator[int]:
    yield 2
    yield 3
    for n in count(5, 6):
        yield n
        yield n + 2


def prime_factors(n: int) -> Iterator[int]:
    if n < 1:
        raise ValueError(f"Expected positive number. Got {n}.")
    ps = pseudoprimes()
    while (p := next(ps)) <= math.sqrt(n):
        while n % p == 0:
            yield p
            n //= p
    if n > 1:
        yield n


def all_factors(n: int) -> Iterator[int]:
    prime_factor_counts = Counter(prime_factors(n))
    pure_counts = prime_factor_counts.values()
    permuted_counts = cross_product(*(range(c + 1) for c in pure_counts))
    for pcs in permuted_counts:
        yield reduce(operator.mul, (b**e for b, e in zip(prime_factor_counts.keys(), pcs)), 1)


def sum_of_factors(n: int) -> int:
    return sum(all_factors(n))


def chinese_remainder_theorem(divisors: Iterable[int], remainders: Iterable[int]) -> int:
    def _iter(dr_left: tuple[int, int], dr_right: tuple[int, int]) -> tuple[int, int]:
        for n in count(dr_left[1], dr_left[0]):
            if n % dr_right[0] == dr_right[1]:
                return dr_left[0] * dr_right[0], n

    divisors_and_remainders = sorted(zip(divisors, remainders), reverse=True)
    return reduce(_iter, divisors_and_remainders)[1]
