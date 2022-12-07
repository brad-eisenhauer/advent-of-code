import logging
from datetime import date
from importlib import import_module
from os import getenv
from pathlib import Path
from typing import Optional

import pytest
import requests
import typer
from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader, select_autoescape

from advent_of_code.base import USER_AGENT, PuzzlePart, Solution
from advent_of_code.util import timer

CURRENT_YEAR = date.today().year
app = typer.Typer(name="aoc")

log = logging.getLogger("aoc")
app_state: dict = {}


@app.callback()
def main(
    debug: bool = typer.Option(False, help="Enable debug logging"),
    input_file: Optional[str] = typer.Option(None, help="Input file name"),
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    app_state["input_file"] = input_file


@app.command(help="Create a stub solution for selected puzzle")
def init(
    day: int = typer.Argument(..., help="Puzzle day to create"),
    year: int = typer.Argument(CURRENT_YEAR, help="Puzzle year to create"),
):
    output_path = Path(__file__).parent / f"aoc{year}" / f"aoc{day:02}.py"
    prep_filesystem(output_path)
    env = Environment(
        loader=PackageLoader("advent_of_code"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("aoc.py.jinja")
    with open(output_path, "w") as f:
        f.write(template.render(day=day, year=year))


def prep_filesystem(module_path: Path):
    if module_path.exists():
        raise OSError("File already exists.")
    pkg_path = module_path.parent
    pkg_init_path = pkg_path / "__init__.py"
    if not pkg_path.exists():
        pkg_path.mkdir(parents=True)
    elif not pkg_path.is_dir():
        raise OSError(f"{pkg_path} exists but is not a directory.")
    if not pkg_init_path.exists():
        pkg_init_path.touch()


@app.command(help="Run unit tests for selected puzzle")
def test(
    day: int = typer.Argument(..., help="Puzzle day to test"),
    year: int = typer.Argument(CURRENT_YEAR, help="Puzzle year to test"),
):
    file = Path(__file__).parent / f"aoc{year}" / f"aoc{day:02}.py"
    pytest.main([file, f"--cov=advent_of_code.aoc{year}.aoc{day:02}"])


@app.command(help="Run solutions for selected puzzle")
def run(
    day: int = typer.Argument(..., help="Puzzle day to run"),
    year: int = typer.Argument(CURRENT_YEAR, help="Puzzle year to run"),
    part: PuzzlePart = typer.Option(PuzzlePart.All, help="Puzzle part to run"),
):
    match part:
        case PuzzlePart.All:
            parts = (PuzzlePart.One, PuzzlePart.Two)
        case _:
            parts = (part,)

    solution = load_solution(day, year)
    for p in parts:
        try:
            with timer():
                result = solution.solve_part(p)
                print(f"Part {p} solution: {result}")
        except NotImplementedError as e:
            print(e)


def load_solution(day: int, year: int) -> Solution:
    module_name = f"advent_of_code.aoc{year}.aoc{day:02}"
    mod = import_module(module_name)

    solution_cls = mod.__dict__.get("AocSolution")
    if solution_cls is None:
        raise TypeError(f"{module_name} contains no implementation of Solution.")

    return solution_cls(input_file=app_state.get("input_file"))


@app.command(help="Submit solution for selected puzzle")
def submit(
    day: int = typer.Argument(..., help="Puzzle day to submit"),
    year: int = typer.Argument(CURRENT_YEAR, help="Puzzle year to submit"),
    part: PuzzlePart = typer.Option(..., help="Puzzle part to submit"),
):
    match part:
        case PuzzlePart.All:
            raise ValueError("Can only submit one puzzle part at a time.")
        case PuzzlePart.One:
            level = 1
        case PuzzlePart.Two:
            level = 2
        case _:
            raise ValueError("Unrecognized PuzzlePart.")
    solution = load_solution(day, year)
    result = solution.solve_part(part)
    load_dotenv()
    url = f"https://adventofcode.com/{year}/day/{day}/answer"
    response = requests.post(
        url,
        data={"level": level, "answer": result},
        cookies={"session": getenv("AOC_SESSION")},
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    content = response.content.decode()
    log.debug("Content: %s", content)
    if "That's the right answer" in content:
        print("That's the right answer!")
    elif "That's not the right answer" in content:
        print("Something went wrong.")
        raise typer.Exit(1)
    else:
        print("Duplicate submission. Please, don't do that again.")
