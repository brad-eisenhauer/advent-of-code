from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")
Vector = tuple[int, ...]


def kdtree(point_list: list[T], accessor: Callable[[T], Vector], depth: int = 0):
    if not point_list:
        return None

    k = len(accessor(point_list[0]))  # assumes all points have the same dimension
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list by axis and choose median as pivot element
    point_list.sort(key=lambda t: accessor(t)[axis])
    median = len(point_list) // 2

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], accessor, depth + 1),
        right_child=kdtree(point_list[median + 1 :], accessor, depth + 1),
    )


@dataclass
class Node(Generic[T]):
    location: T
    left_child: Optional[Node[T]]
    right_child: Optional[Node[T]]


def find_nearest_neighbor(
    loc: Vector, tree: Node[T], accessor: Callable[[T], Vector], depth: int = 0
) -> T:
    axis = depth % len(loc)
    split = accessor(tree.location)[axis]

    if loc[axis] < split:
        if tree.right_child is None:
            best_estimate = tree.location
        else:
            best_estimate = find_nearest_neighbor(loc, tree.right_child, accessor, depth + 1)
        other_side = tree.left_child
    else:
        if tree.left_child is None:
            best_estimate = tree.location
        else:
            best_estimate = find_nearest_neighbor(loc, tree.left_child, accessor, depth + 1)
        other_side = tree.right_child

    if other_side is None:
        return best_estimate

    best_distance = sum(abs(a - b) for a, b in zip(loc, accessor(best_estimate)))
    if other_side is not None and abs(loc[axis] - split) <= best_distance:
        alternate = find_nearest_neighbor(loc, other_side, accessor, depth + 1)
        alternate_distance = sum(abs(a - b) for a, b in zip(loc, accessor(alternate)))
        if alternate_distance < best_distance:
            return alternate

    return best_estimate
