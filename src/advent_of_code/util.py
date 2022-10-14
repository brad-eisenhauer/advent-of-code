import logging
import time
from abc import abstractmethod
from contextlib import contextmanager
from heapq import heappop, heappush
from itertools import islice, tee
from typing import (
    Callable,
    ContextManager,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
)

import networkx as nx

log = logging.getLogger("aoc")

N = TypeVar("N")
T = TypeVar("T")
S = TypeVar("S")


def partition(
    predicate: Callable[[T], bool], items: Iterable[T]
) -> tuple[Iterator[T], Iterator[T]]:
    """
    Split items into two sequences, based on some predicate condition

    Parameters
    ----------
    predicate
        Boolean function by which to categorize items
    items
        Items to be categorized

    Returns
    -------
    Iterators of failing and passing items, respectively
    """
    tested_items = ((item, predicate(item)) for item in items)
    failed, passed = tee(tested_items)
    return (item for item, result in failed if not result), (
        item for item, result in passed if result
    )


def greatest_common_divisor(a: int, b: int) -> int:
    if b == 0:
        return abs(a)
    return greatest_common_divisor(b, a % b)


def least_common_multiple(a: int, b: int) -> int:
    return a // greatest_common_divisor(a, b) * b


@contextmanager
def timer():
    start = time.monotonic()
    try:
        yield
    finally:
        end = time.monotonic()
        print(f"Elapsed time: {(end - start) * 1000} ms")


def make_sequence(items: Iterable[T]) -> Sequence[T]:
    if isinstance(items, Sequence):
        return items
    return tuple(items)


def create_windows(items: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    iterators = tee(items, n)
    offset_iterators = (islice(iterator, offset, None) for offset, iterator in enumerate(iterators))
    return zip(*offset_iterators)


class Timer(ContextManager):
    def __init__(self):
        self.start: float = 0.0
        self.last_check: Optional[float] = None
        self.check_index = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.check("Elapsed time")

    def check(self, label: Optional[str] = None):
        check_time = time.time()
        self.check_index += 1
        message = (
            f"[{self.check_index}] {label or 'Time check'}: "
            f"{self.get_formatted_time(check_time)}"
        )
        message += self.get_last_check_msg(check_time)
        print(message)
        self.last_check = check_time

    def get_formatted_time(self, end: float, start: Optional[float] = None) -> str:
        start = start or self.start
        return f"{(end - start) * 1000:0.3f} ms"

    def get_last_check_msg(self, check_time: float) -> str:
        if self.last_check is None:
            return ""
        return f" ({self.get_formatted_time(check_time, self.last_check)} since last check)"


class PriorityQueue(Generic[T]):
    def __init__(self):
        self._contents: list[T] = []

    def __bool__(self):
        return bool(self._contents)

    def __len__(self):
        return len(self._contents)

    def peek(self) -> T:
        return self._contents[0]

    def pop(self) -> T:
        _, result = heappop(self._contents)
        return result

    def push(self, priority: int, item: T):
        heappush(self._contents, (priority, item))


class AStar(Generic[S]):
    """
    A* implementation based on https://www.redblobgames.com/pathfinding/a-star/introduction.html
    """
    @abstractmethod
    def is_goal_state(self, state: S) -> bool:
        ...

    @abstractmethod
    def generate_next_states(self, state: S) -> Iterator[tuple[int, S]]:
        """For a given state, generate successor states and the transition cost."""
        ...

    def heuristic(self, state: S) -> int:
        """With no heuristic, A* is equivalent to Djikstra."""
        return 0

    def _find_min_cost_path(self, initial_state: S) -> tuple[Optional[S], dict[S, int], dict[S, Optional[S]]]:
        accumulated_cost: dict[S, int] = {initial_state: 0}
        came_from: dict[S, Optional[S]] = {initial_state: None}
        frontier: PriorityQueue[S] = PriorityQueue()
        frontier.push(0, initial_state)
        state_count = 0
        while frontier:
            current_state = frontier.pop()
            state_count += 1
            if self.is_goal_state(current_state):
                log.debug("Evaluated %d states.", state_count)
                return current_state, accumulated_cost, came_from
            current_cost = accumulated_cost[current_state]
            for cost, next_state in self.generate_next_states(current_state):
                total_cost = current_cost + cost
                if next_state not in accumulated_cost or total_cost < accumulated_cost[next_state]:
                    accumulated_cost[next_state] = total_cost
                    came_from[next_state] = current_state
                    frontier.push(total_cost + self.heuristic(next_state), next_state)
        raise ValueError("Goal state not found.")

    def find_min_cost_path(self, initial_state: S) -> Iterator[tuple[S, int]]:
        goal_state, accumulated_cost, came_from = self._find_min_cost_path(initial_state)
        backtrack = []
        state = goal_state
        while state is not None:
            backtrack.append((state, accumulated_cost[state]))
            state = came_from[state]
        return reversed(backtrack)

    def find_min_cost_to_goal(self, initial_state: S) -> int:
        goal_state, accumulated_cost, _ = self._find_min_cost_path(initial_state)
        return accumulated_cost[goal_state]


class GraphSimplifier(Generic[N]):
    DEAD_END = 1
    HALLWAY = 2

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @abstractmethod
    def is_protected(self, node: N, mode: int) -> bool:
        ...

    def simplify(self, depth: int = 1):
        g = self.graph
        nodes_to_remove = []
        for node in g.nodes:
            neighbors = list(g.neighbors(node))
            match len(neighbors):
                case 1:  # dead end
                    if self.is_protected(node, self.DEAD_END):
                        continue
                    nodes_to_remove.append(node)
                    g.remove_edge(node, neighbors[0])
                case 2:  # hallway
                    if self.is_protected(node, self.HALLWAY):
                        continue
                    nodes_to_remove.append(node)
                    weight = sum(g.get_edge_data(node, n)["weight"] for n in neighbors)
                    for n in neighbors:
                        g.remove_edge(node, n)
                    g.add_edge(*neighbors, weight=weight)
                case _:
                    ...
        if nodes_to_remove:
            g.remove_nodes_from(nodes_to_remove)
            self.simplify(depth + 1)
        else:
            log.debug("Ran simplify %d times.", depth)
            log.debug("Simlified graph has %d nodes.", len(g.nodes))
