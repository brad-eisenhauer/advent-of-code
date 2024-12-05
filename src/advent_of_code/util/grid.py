from typing import TypeVar

T = TypeVar("T")


def pad_grid(grid: list[list[T]], value: T, size: int) -> list[list[T]]:
    orig_width = max(len(line) for line in grid)
    new_width = orig_width + 2 * size

    result = [[value] * new_width] * size
    for line in grid:
        result.append([value] * size + line + [value] * (new_width - len(line) - size))
    result.extend([[value] * new_width] * size)
    return result


def pad_str_grid(grid: list[str], value: str, size: int) -> list[str]:
    orig_width = max(len(line) for line in grid)
    new_width = orig_width + 2 * size
    result = [value * new_width] * size
    for line in grid:
        result.append(value * size + line + value * (new_width - len(line) - size))
    result.extend([value * new_width] * size)
    return result
