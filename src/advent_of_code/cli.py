from datetime import date
from importlib import import_module
from os import getenv
from pathlib import Path

import pytest
import requests
import typer
from dotenv import load_dotenv

from advent_of_code.base import PuzzlePart, Solution
from advent_of_code.util import timer

CURRENT_YEAR = date.today().year
app = typer.Typer(name="aoc")


@app.command(help="Run unit tests for selected puzzle")
def test(
    day: int = typer.Argument(..., help="Puzzle day to test"),
    year: int = typer.Argument(CURRENT_YEAR, help="Puzzle year to test"),
):
    file = Path(__file__).parent / f"aoc{year}" / f"aoc{day:02}.py"
    pytest.main(file)


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

    for key, item in mod.__dict__.items():
        if key == "AocSolution":
            solution_cls = item
            break
    else:
        raise TypeError(f"{module_name} contains no implementation of Solution.")

    return solution_cls()


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
        url, data={"level": level, "answer": result}, cookies={"session": getenv("AOC_SESSION")}
    )
    response.raise_for_status()
    content = response.content.decode()
    if "That's the right answer" in content:
        print("That's the right answer!")
    elif "That's not the right answer" in content:
        print("Something went wrong.")
        raise typer.Exit(1)
    else:
        print("Duplicate submission. Please, don't do that again.")
