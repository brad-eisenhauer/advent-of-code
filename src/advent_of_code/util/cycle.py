from itertools import count
from typing import Iterator, TypeVar

T = TypeVar("T")


def detect_cycle(items: Iterator[T]) -> tuple[int, int] | None:
    record: list[T] = [next(items)]
    for item in items:
        if item == record[len(record) // 2]:
            detected_cycle_start = len(record) // 2
            detected_cycle_len = len(record) - detected_cycle_start
            min_cycle_len = detected_cycle_len
            record.append(item)
            for n in count(2):
                if n * n > detected_cycle_len:
                    break
                if (
                    detected_cycle_len % n == 0
                    and (
                        record[detected_cycle_start]
                        == record[detected_cycle_start + detected_cycle_len // n]
                    )
                ):
                    min_cycle_len = detected_cycle_len // n
            break
        record.append(item)
    else:
        return None

    for index, item in enumerate(record):
        if index + min_cycle_len < len(record) and item == record[index + min_cycle_len]:
            return min_cycle_len, index + min_cycle_len
