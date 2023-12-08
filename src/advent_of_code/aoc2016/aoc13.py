"""Advent of Code 2016, day 13: https://adventofcode.com/2016/day/13"""
from __future__ import annotations

from collections import deque
from typing import Iterator

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(13, 2016, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as fp:
            favorite_value = int(fp.read().strip())
        navigator = Navigator(favorite_value)
        return navigator.find_min_cost_to_goal((1, 1))

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            favorite_value = int(fp.read().strip())
        bfs = BFS(favorite_value)
        return sum(1 for _ in bfs.visit_to_depth((1, 1), 50))


Vector = tuple[int, ...]
directions: list[Vector] = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class Navigator(AStar[Vector]):
    def __init__(self, favorite_value: int, destination: Vector = (31, 39)):
        self.favorite_value = favorite_value
        self.destination = destination

    def is_goal_state(self, state: Vector) -> bool:
        return state == self.destination

    def heuristic(self, state: Vector) -> int:
        return sum(abs(a - b) for a, b in zip(state, self.destination))

    def generate_next_states(self, state: Vector) -> Iterator[tuple[int, Vector]]:
        for d in directions:
            next_loc = tuple(a + b for a, b in zip(state, d))
            if not self.is_wall(next_loc):
                yield 1, next_loc

    def is_wall(self, state: Vector) -> bool:
        if any(c < 0 for c in state):
            return True
        x, y = state
        discriminant = x * x + 3 * x + 2 * x * y + y + y * y + self.favorite_value
        return density(discriminant) % 2 > 0


def density(n: int) -> int:
    result = 0
    while n > 0:
        result += n & 1
        n >>= 1
    return result


class BFS:
    def __init__(self, favorite_value: int):
        self.navigator = Navigator(favorite_value)

    def visit_to_depth(self, initial_loc: Vector, depth: int) -> Iterator[Vector]:
        visited: set[Vector] = {initial_loc}
        frontier: deque[tuple[int, Vector]] = deque()
        frontier.append((0, initial_loc))

        while frontier:
            current_depth, current_loc = frontier.popleft()
            yield current_loc
            next_depth = current_depth + 1
            if next_depth > depth:
                continue
            for _, loc in self.navigator.generate_next_states(current_loc):
                if loc in visited:
                    continue
                visited.add(loc)
                frontier.append((next_depth, loc))
