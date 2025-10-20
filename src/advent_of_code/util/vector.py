"""Vectors and operations"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import ClassVar, Iterable, Iterator, Optional, Self, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Vector(Iterable[int]):
    coords: tuple[int, ...]

    @classmethod
    def from_iterable(cls, coords: Iterable[int]) -> Self:
        return cls(tuple(coords))

    def __add__(self, other) -> Self:
        return self._binary_op(operator.add, other)

    def is_bounded_by(self, upper_limit: Vector, lower_limit: Optional[Vector] = None) -> bool:
        if lower_limit is None:
            lower_limit = self.origin()
        return all(x <= y < z for x, y, z in zip(lower_limit, self, upper_limit))

    def origin(self) -> Self:
        return type(self)((0,) * len(self))

    def __mod__(self, other) -> Self:
        return self._binary_op(operator.mod, other)

    def _binary_op(self, op, other) -> Self:
        if not isinstance(other, Vector):
            raise TypeError()
        if len(self) != len(other):
            raise ValueError("Cannot operate on vectors of different dimensions.")
        return type(self).from_iterable(op(a, b) for a, b in zip(self, other))

    def __iter__(self) -> Iterator[int]:
        return iter(self.coords)

    def __len__(self) -> int:
        return len(self.coords)


class Vector2(Vector):
    coords: tuple[int, int]

    _NEIGHBORS: ClassVar[Optional[list[Vector2]]] = None

    @classmethod
    def neighbors(cls) -> Iterator[Vector2]:
        if cls._NEIGHBORS is None:
            cls._NEIGHBORS = [Vector2(cs) for cs in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        return iter(cls._NEIGHBORS)

    def __len__(self) -> int:
        return 2

    def extract(self, data: Sequence[Sequence[T]]) -> T:
        return data[self.coords[0]][self.coords[1]]
