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

log = logging.getLogger("aoc")

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
        return heappop(self._contents)

    def push(self, priority: int, item: T):
        heappush(self._contents, (priority, item))


class Dijkstra(Generic[S]):
    @abstractmethod
    def is_goal_state(self, state: S) -> bool:
        ...

    @abstractmethod
    def generate_next_states(self, state: S) -> Iterator[tuple[int, S]]:
        ...

    def find_min_cost_path(self, initial_state: S) -> Iterator[tuple[S, int]]:
        goal_state: Optional[S] = None
        costs: dict[S, int] = {initial_state: 0}
        came_from: dict[S, Optional[S]] = {initial_state: None}
        frontier: PriorityQueue[S] = PriorityQueue()
        frontier.push(0, initial_state)
        while frontier:
            current_cost, current_state = frontier.pop()
            if goal_state and current_cost > costs[goal_state]:
                break
            for cost, next_state in self.generate_next_states(current_state):
                total_cost = current_cost + cost
                if self.is_goal_state(next_state) and (goal_state is None or total_cost < costs[goal_state]):
                    goal_state = next_state
                if next_state not in costs or total_cost < costs[next_state]:
                    costs[next_state] = total_cost
                    came_from[next_state] = current_state
                    frontier.push(total_cost, next_state)
        result = []
        state = goal_state
        while state is not None:
            result.append((state, costs[state]))
            state = came_from[state]

        return reversed(result)

    def find_min_cost_to_goal(self, initial_state: S) -> int:
        result = -1
        for _, result in self.find_min_cost_path(initial_state):
            ...
        if result < 0:
            raise ValueError("No result found.")
        return result
