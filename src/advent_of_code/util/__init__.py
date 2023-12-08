import time
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Callable, ContextManager, Iterable, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T")


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


def make_sequence(items: Iterable[T]) -> Sequence[T]:
    if isinstance(items, Sequence):
        return items
    return tuple(items)


def create_windows(items: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    iterators = tee(items, n)
    offset_iterators = (islice(iterator, offset, None) for offset, iterator in enumerate(iterators))
    return zip(*offset_iterators)


def create_groups(items: Iterable, n: int, zip_all: bool = False, fill_value=None) -> Iterator:
    zipper = partial(zip_longest, fill_value=fill_value) if zip_all else zip
    args = [iter(items)] * n
    return zipper(*args)


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
