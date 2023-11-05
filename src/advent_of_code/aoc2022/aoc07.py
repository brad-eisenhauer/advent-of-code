"""Advent of Code 2022, day 7: https://adventofcode.com/2022/day/7"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from io import StringIO
from itertools import chain
from typing import Callable, Iterable, Optional, Protocol, TextIO, TypeVar

import pytest

from advent_of_code.base import Solution

T = TypeVar("T")

TOTAL_STORAGE = 70_000_000


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(7, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            fs = explore(f)
        return fs.sum_of_small_dir_sizes()

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            fs = explore(f)
        available_storage = TOTAL_STORAGE - fs.size
        required_storage = 30_000_000 - available_storage
        result = fs.find_smallest_dir_greater_than(required_storage)
        return result.size


class FileSystemObject(Protocol):
    @property
    def size(self) -> int:
        ...

    def fold(self, f: Callable[[FileSystemObject], T], r: Callable[[T, ...], T]) -> T:
        ...


@dataclass
class File:
    name: str
    size: int

    def fold(self, f: Callable[[FileSystemObject], T], _r: Callable[[T, ...], T]) -> T:
        return f(self)


@dataclass
class Directory:
    name: str
    parent: Optional[Directory] = field(default=None)
    contents: dict[str, FileSystemObject] = field(default_factory=dict)

    def __eq__(self, other):
        return self.name == self.name and self.contents == self.contents

    @cached_property
    def size(self) -> int:
        return sum(item.size for item in self.contents.values())

    def fold(self, f: Callable[[FileSystemObject], T], r: Callable[[Iterable[T]], T]) -> T:
        return r(chain((f(self),), (item.fold(f, r) for item in self.contents.values())))

    def sum_of_small_dir_sizes(self, limit: int = 100000) -> int:
        return self.fold(
            lambda item: item.size if isinstance(item, Directory) and item.size <= limit else 0, sum
        )

    def find_smallest_dir_greater_than(self, limit: int) -> Optional[Directory]:
        def min_size(dirs: Iterable[Optional[Directory]]) -> Optional[Directory]:
            try:
                return min((d for d in dirs if d is not None), key=lambda d: d.size)
            except ValueError:
                return None

        return self.fold(
            lambda item: item if isinstance(item, Directory) and item.size >= limit else None,
            min_size,
        )


def explore(terminal: TextIO) -> Directory:
    current_dir: Optional[Directory] = None

    def find_root_dir():
        dir = current_dir
        while dir.parent is not None:
            dir = dir.parent
        return dir

    for line in terminal:
        tokens = line.split()
        match tokens:
            case ["$", "cd", "/"]:
                current_dir = Directory("/") if current_dir is None else find_root_dir()
            case ["$", "cd", ".."]:
                current_dir = current_dir.parent
            case ["$", "cd", dir_name]:
                sub_dir = current_dir.contents[dir_name]
                assert isinstance(sub_dir, Directory)
                current_dir = sub_dir
            case ["$", "ls"]:
                ...
            case ["dir", dir_name]:
                current_dir.contents[dir_name] = Directory(dir_name, parent=current_dir)
            case [file_size, file_name]:
                current_dir.contents[file_name] = File(file_name, int(file_size))

    return find_root_dir()


SAMPLE_INPUTS = [
    """\
$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k
""",
]


@pytest.fixture()
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


@pytest.fixture()
def file_system(sample_input) -> Directory:
    return explore(sample_input)


def test_explore(file_system):
    expected = Directory(
        name="/",
        contents={
            "a": Directory(
                name="a",
                contents={
                    "e": Directory("e", contents={"i": File("i", 584)}),
                    "f": File("f", 29116),
                    "g": File("g", 2557),
                    "h.lst": File("h.lst", 62596),
                },
            ),
            "b.txt": File("b.txt", 14848514),
            "c.dat": File("c.dat", 8504156),
            "d": Directory(
                name="d",
                contents={
                    "j": File("j", 4060174),
                    "d.log": File("d.log", 8033020),
                    "d.ext": File("d.ext", 5626152),
                    "k": File("k", 7214296),
                },
            ),
        },
    )

    assert file_system == expected


def test_sum_of_small_directory_sizes(file_system):
    assert file_system.sum_of_small_dir_sizes() == 95437


def test_free_up_space(file_system):
    available_storage = TOTAL_STORAGE - file_system.size
    required_storage = 30_000_000 - available_storage
    result = file_system.find_smallest_dir_greater_than(required_storage)
    assert result.name == "d"
