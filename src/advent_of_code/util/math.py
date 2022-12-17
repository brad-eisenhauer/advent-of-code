from typing import Optional


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


def intersect_ranges(left: range, right: range) -> range:
    inter_min = max(min(left), min(right))
    inter_max = min(max(left), max(right))
    return range(inter_min, inter_max + 1)


def union_ranges(left: range, right: range) -> Optional[range]:
    match intersect_ranges(left, right):
        case range(a, b) if b >= a - 1:
            union_min = min(min(left), min(right))
            union_max = max(max(left), max(right))
            return range(union_min, union_max + 1)
        case _:
            return None
