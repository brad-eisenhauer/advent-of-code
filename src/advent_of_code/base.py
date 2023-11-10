from contextlib import AbstractContextManager
from datetime import date
from enum import Enum
from os import getenv
from pathlib import Path
from typing import IO, Generic, Optional, TypeVar, Union

import requests
from dotenv import load_dotenv

USER_AGENT = "github.com/brad-eisenhauer/advent-of-code by bradley.eisenhauer@gmail.com"

T = TypeVar("T")
U = TypeVar("U")


class PuzzlePart(str, Enum):
    All = "all"
    One = "one"
    Two = "two"


class Solution(Generic[T, U]):
    def __init__(self, day: int, year: int, input_file: Optional[str] = None):
        self.day = day
        self.year = year
        self.input_file = input_file

    def open_input(self) -> AbstractContextManager[IO]:
        path = self._get_input_path()
        return open(path)

    def solve_part(self, part: PuzzlePart) -> Union[T, U]:
        match part:
            case PuzzlePart.All:
                raise ValueError("Part argument must be 'one' or 'two'.")
            case PuzzlePart.One:
                return self.solve_part_one()
            case PuzzlePart.Two:
                return self.solve_part_two()
            case _:
                raise ValueError("Unrecognized part.")

    def solve_part_one(self) -> T:
        raise NotImplementedError("Puzzle part one not yet implemented.")

    def solve_part_two(self) -> U:
        raise NotImplementedError("Puzzle part two not yet implemented.")

    def _get_input_path(self) -> Path:
        resources_path = Path() / "input" / str(self.year)
        input_path = resources_path / (
            self.input_file if self.input_file is not None else f"input{self.day:02d}.txt"
        )

        if not input_path.exists():
            if self.input_file is not None:
                raise FileNotFoundError(
                    f"File {self.input_file} does not exist in {resources_path}."
                )
            input_path.parent.mkdir(parents=True, exist_ok=True)
            download_input(input_path, self.day, self.year or date.today().year)

        return input_path


def download_input(download_path: Path, day: int, year: int):
    load_dotenv()
    download_url = f"https://adventofcode.com/{year}/day/{day}/input"
    response = requests.get(
        download_url,
        cookies={"session": getenv("AOC_SESSION")},
        headers={"User-Agent": USER_AGENT},
        timeout=3.05,
    )
    response.raise_for_status()
    with open(download_path, "w") as fp:
        fp.write(response.content.decode())
