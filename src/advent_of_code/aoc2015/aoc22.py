"""Advent of Code 2015, day 22: https://adventofcode.com/2015/day/22"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Iterator

import yaml

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar

log = logging.getLogger("aoc")


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(22, 2015, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            boss_stats = yaml.safe_load(f)
        initial_state = BattleState(
            player_hp=50,
            player_mana=500,
            boss_hp=boss_stats["Hit Points"],
            boss_damage=boss_stats["Damage"],
        )
        solver = Battler()
        result = solver.find_min_cost_to_goal(initial_state)
        return result

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            boss_stats = yaml.safe_load(f)
        initial_state = BattleState(
            player_hp=49,
            player_mana=500,
            boss_hp=boss_stats["Hit Points"],
            boss_damage=boss_stats["Damage"],
            hard_mode=True,
        )
        solver = Battler()
        if log.isEnabledFor(logging.DEBUG):
            for state, cost in solver.find_min_cost_path(initial_state):
                log.debug("%d: %s", cost, state)
        result = solver.find_min_cost_to_goal(initial_state)
        return result


class Effect(Enum):
    Shield = auto()
    Poison = auto()
    Recharge = auto()


class Spell(Enum):
    def __new__(cls, value, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value, mana_cost, effect):
        self.mana_cost = mana_cost
        self.effect = effect

    MagicMissile = auto(), 53, None
    Drain = auto(), 73, None
    Shield = auto(), 113, Effect.Shield
    Poison = auto(), 173, Effect.Poison
    Recharge = auto(), 229, Effect.Recharge


@dataclass(frozen=True)
class ActiveEffect:
    effect: Effect
    turns_remaining: int


@dataclass(frozen=True)
class BattleState:
    player_hp: int
    player_mana: int
    boss_hp: int
    boss_damage: int
    active_effects: frozenset[ActiveEffect] = field(default_factory=frozenset)
    hard_mode: bool = field(default=False)

    @property
    def player_defense(self) -> int:
        return 7 if any(f.effect is Effect.Shield for f in self.active_effects) else 0

    def __lt__(self, other) -> bool:
        return self.player_hp < other.player_hp

    def apply_effects(self) -> BattleState:
        result = vars(self)
        effects = set()
        for effect in self.active_effects:
            match effect.effect:
                case Effect.Poison:
                    result["boss_hp"] -= 3
                case Effect.Recharge:
                    result["player_mana"] += 101
            if effect.turns_remaining > 1:
                effects.add(replace(effect, turns_remaining=effect.turns_remaining - 1))
        return replace(
            self,
            player_hp=result["player_hp"],
            player_mana=result["player_mana"],
            boss_hp=result["boss_hp"],
            active_effects=frozenset(effects),
        )

    def can_cast(self, spell: Spell) -> bool:
        if self.player_hp <= 0:
            return False
        if spell.mana_cost > self.player_mana:
            return False
        if spell.effect is not None and any(f.effect is spell.effect for f in self.active_effects):
            return False
        return True

    def cast(self, spell: Spell) -> BattleState:
        player_mana = self.player_mana - spell.mana_cost
        match spell:
            case Spell.MagicMissile:
                return replace(self, player_mana=player_mana, boss_hp=self.boss_hp - 4)
            case Spell.Drain:
                return replace(
                    self,
                    player_mana=player_mana,
                    player_hp=self.player_hp + 2,
                    boss_hp=self.boss_hp - 2,
                )
            case Spell.Shield:
                return replace(
                    self,
                    player_mana=player_mana,
                    active_effects=self.active_effects | {ActiveEffect(Effect.Shield, 6)},
                )
            case Spell.Poison:
                return replace(
                    self,
                    player_mana=player_mana,
                    active_effects=self.active_effects | {ActiveEffect(Effect.Poison, 6)},
                )
            case Spell.Recharge:
                return replace(
                    self,
                    player_mana=player_mana,
                    active_effects=self.active_effects | {ActiveEffect(Effect.Recharge, 5)},
                )

    def boss_attack(self) -> BattleState:
        if self.boss_hp <= 0:
            return self
        boss_damage = max(1, self.boss_damage - self.player_defense)
        return replace(self, player_hp=self.player_hp - boss_damage)

    def round(self, spell: Spell) -> BattleState:
        state = self
        state = state.cast(spell)
        if state.boss_hp <= 0 or state.player_hp <= 0:
            return state
        state = state.apply_effects()
        state = state.boss_attack()
        if state.hard_mode and state.boss_hp > 0:
            state = replace(state, player_hp=state.player_hp - 1)
        state = state.apply_effects()
        return state


class Battler(AStar[BattleState]):
    def is_goal_state(self, state: BattleState) -> bool:
        return state.boss_hp <= 0

    def generate_next_states(self, state: BattleState) -> Iterator[tuple[int, BattleState]]:
        for spell in Spell:
            if state.can_cast(spell):
                yield spell.mana_cost, state.round(spell)

    def heuristic(self, state: BattleState) -> int:
        min_mana = 173 * (state.boss_hp // 18)
        min_mana += min(173, 53 * math.ceil((state.boss_hp % 18) / 4))
        return min_mana
