"""Advent of Code 2015, day 21: https://adventofcode.com/2015/day/21"""
from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass
from itertools import chain, combinations, product
from typing import Callable, Iterator, Optional

import yaml

from advent_of_code.base import Solution

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(21, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            boss_stats = yaml.safe_load(f)
        return calc_optimal_gear(boss_stats)

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            boss_stats = yaml.safe_load(f)
        return calc_optimal_gear(boss_stats, comp=operator.gt, winner="boss")


@dataclass
class Item:
    cost: int = 0
    damage: int = 0
    armor: int = 0


WEAPONS = {
    "Dagger": Item(8, damage=4),
    "Shortsword": Item(10, damage=5),
    "Warhammer": Item(25, damage=6),
    "Longsword": Item(40, damage=7),
    "Greataxe": Item(74, damage=8),
}

ARMOR = {
    "None": Item(0, armor=0),
    "Leather": Item(13, armor=1),
    "Chainmail": Item(31, armor=2),
    "Splintmail": Item(53, armor=3),
    "Bandedmail": Item(75, armor=4),
    "Platemail": Item(102, armor=5),
}

RINGS = {
    "Damage +1": Item(25, damage=1),
    "Damage +2": Item(50, damage=2),
    "Damage +3": Item(100, damage=3),
    "Defense +1": Item(20, armor=1),
    "Defense +2": Item(40, armor=2),
    "Defense +3": Item(80, armor=3),
}


def shop() -> Iterator[list[Item]]:
    ring_options: list[tuple[Item]] = list(
        chain.from_iterable(combinations(RINGS.values(), n) for n in range(3))
    )
    for weapon, armor, rings in product(WEAPONS.values(), ARMOR.values(), ring_options):
        yield [weapon, armor, *rings]


def calc_stats(items: list[Item], hp: int = 100) -> tuple[int, dict[str, int]]:
    result = Item()
    for item in items:
        result.cost += item.cost
        result.damage += item.damage
        result.armor += item.armor
    return result.cost, {"Hit Points": hp, "Damage": result.damage, "Armor": result.armor}


def decide_winner(player: dict[str, int], boss: dict[str, int]) -> str:
    log.debug("Fight start: player=%s, boss=%s", player, boss)
    player_damage = max(1, player["Damage"] - boss["Armor"])
    boss_damage = max(1, boss["Damage"] - player["Armor"])
    player_turns = math.ceil(boss["Hit Points"] / player_damage)
    boss_turns = math.ceil(player["Hit Points"] / boss_damage)
    return "player" if player_turns <= boss_turns else "boss"


def calc_optimal_gear(
    boss_stats: dict[str, int],
    comp: Callable[[int, int], bool] = operator.lt,
    winner: str = "player",
) -> Optional[int]:
    opt_cost: Optional[int] = None
    for gear in shop():
        cost, stats = calc_stats(gear)
        if (opt_cost is None or comp(cost, opt_cost)) and decide_winner(stats, boss_stats) == winner:
            opt_cost = cost
            log.debug("gear=%s, cost=%d", gear, cost)
    return opt_cost


def test_shop():
    assert sum(1 for _ in shop()) == (5 * 6) * (1 + 6 + 15)
