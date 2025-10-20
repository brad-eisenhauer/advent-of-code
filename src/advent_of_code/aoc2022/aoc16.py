"""Advent of Code 2022, day 16: https://adventofcode.com/2022/day/16"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from io import StringIO
from itertools import product
from typing import Iterator, TextIO, TypeVar

import networkx as nx
import pytest

from advent_of_code.base import Solution
from advent_of_code.util.pathfinder import AStar


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(16, 2022, **kwargs)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            graph = parse_valves(f)
        return find_max_release(Navigator(graph), State(Valve("AA", 0)))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            graph = parse_valves(f)
        return find_max_release(Navigator2(graph), State2((Valve("AA", 0), Valve("AA", 0))))


@dataclass(frozen=True)
class Valve:
    id: str
    flow_rate: int

    def __lt__(self, other):
        return self.id < other.id


def parse_valves(f: TextIO) -> nx.Graph:
    pattern = re.compile(
        r"Valve ([A-Z]+) has flow rate=(\d+); tunnels? leads? to valves? ((?:(?:, )?[A-Z]+)+)"
    )
    valves: dict[str, Valve] = {}
    connects_to: dict[str, list[str]] = {}

    for line in f:
        valve_id, rate, tunnels = pattern.match(line).groups()
        valves[valve_id] = Valve(valve_id, int(rate))
        connects_to[valve_id] = tunnels.split(", ")

    result = nx.Graph()
    result.add_nodes_from(valves.values())
    for valve_id, connections in connects_to.items():
        valve = valves[valve_id]
        for connection_id in connections:
            connected_valve = valves[connection_id]
            result.add_edge(valve, connected_valve)

    return result


@dataclass(frozen=True)
class State:
    current_valve: Valve
    time_remaining: int = field(default=30, compare=False)
    opened_valves: frozenset[Valve] = frozenset()
    accumulated_pressure_release: int = field(default=0, compare=False)

    def __lt__(self, other):
        if self.time_remaining != other.time_remaining:
            return self.time_remaining < other.time_remaining
        if self.current_valve != other.current_valve:
            return self.current_valve < other.current_valve
        return False


class Navigator(AStar[State]):
    def __init__(self, graph: nx.Graph):
        self._graph = graph
        self._max_pressure_release = 30 * max(n.flow_rate for n in graph.nodes)

    def is_goal_state(self, state: State) -> bool:
        if all(v in state.opened_valves for v in self._graph.nodes if v.flow_rate > 0):
            return True
        return state.time_remaining == 0

    def generate_next_states(self, state: State) -> Iterator[tuple[int, State]]:
        time_remaining = state.time_remaining - 1
        potential_release = sum(
            v.flow_rate for v in self._graph.nodes if v not in state.opened_valves
        )
        for neighbor in self._graph.neighbors(state.current_valve):
            yield (
                potential_release,
                replace(state, current_valve=neighbor, time_remaining=time_remaining),
            )
        if state.current_valve.flow_rate > 0 and state.current_valve not in state.opened_valves:
            yield (
                potential_release - state.current_valve.flow_rate,
                State(
                    state.current_valve,
                    time_remaining,
                    state.opened_valves | {state.current_valve},
                    state.accumulated_pressure_release
                    + time_remaining * state.current_valve.flow_rate,
                ),
            )


@dataclass(frozen=True)
class State2:
    current_valves: tuple[Valve, Valve]
    opened_valves: frozenset[Valve] = frozenset()
    time_remaining: int = field(default=26, compare=False)
    accumulated_pressure_release: int = field(default=0, compare=False)

    def __lt__(self, other):
        if self.time_remaining != other.time_remaining:
            return self.time_remaining < other.time_remaining
        if self.current_valves != other.current_valves:
            return self.current_valves < other.current_valves
        return False


class Navigator2(AStar[State2]):
    def __init__(self, graph: nx.Graph):
        self._graph = graph
        self._max_pressure_release = 52 * max(n.flow_rate for n in graph.nodes)

    def is_goal_state(self, state: State2) -> bool:
        if all(v in state.opened_valves for v in self._graph.nodes if v.flow_rate > 0):
            return True
        return state.time_remaining == 0

    def generate_next_states(self, state: State2) -> Iterator[tuple[int, State2]]:
        v1, v2 = state.current_valves
        time_remaining = state.time_remaining - 1
        potential_release = sum(
            v.flow_rate for v in self._graph.nodes if v not in state.opened_valves
        )
        # both move
        for n1, n2 in product(self._graph.neighbors(v1), self._graph.neighbors(v2)):
            yield (
                potential_release * time_remaining,
                replace(state, current_valves=(n1, n2), time_remaining=time_remaining),
            )
        # first node opens, if possible; second moves
        if v1.flow_rate > 0 and v1 not in state.opened_valves:
            for n2 in self._graph.neighbors(v2):
                yield (
                    time_remaining * (potential_release - v1.flow_rate),
                    State2(
                        current_valves=(v1, n2),
                        opened_valves=state.opened_valves | {v1},
                        time_remaining=time_remaining,
                        accumulated_pressure_release=state.accumulated_pressure_release
                        + time_remaining * v1.flow_rate,
                    ),
                )
        # second node opens, if possible; first moves
        if v1 != v2 and v2.flow_rate > 0 and v2 not in state.opened_valves:
            for n1 in self._graph.neighbors(v1):
                yield (
                    time_remaining * (potential_release - v2.flow_rate),
                    State2(
                        current_valves=(n1, v2),
                        opened_valves=state.opened_valves | {v2},
                        time_remaining=time_remaining,
                        accumulated_pressure_release=state.accumulated_pressure_release
                        + time_remaining * v2.flow_rate,
                    ),
                )
        # both open
        if (
            v1 != v2
            and v1.flow_rate > 0
            and v2.flow_rate > 0
            and v1 not in state.opened_valves
            and v2 not in state.opened_valves
        ):
            yield (
                time_remaining * (potential_release - v1.flow_rate - v2.flow_rate),
                State2(
                    current_valves=state.current_valves,
                    opened_valves=state.opened_valves | {v1, v2},
                    time_remaining=time_remaining,
                    accumulated_pressure_release=state.accumulated_pressure_release
                    + time_remaining * (v1.flow_rate + v2.flow_rate),
                ),
            )


T = TypeVar("T", State, State2)


def find_max_release(nav: AStar[T], initial_state: T) -> int:
    for state, _ in nav.find_min_cost_path(initial_state):  # noqa: B007
        ...
    return state.accumulated_pressure_release


SAMPLE_INPUTS = [
    """\
Valve AA has flow rate=0; tunnels lead to valves DD, II, BB
Valve BB has flow rate=13; tunnels lead to valves CC, AA
Valve CC has flow rate=2; tunnels lead to valves DD, BB
Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE
Valve EE has flow rate=3; tunnels lead to valves FF, DD
Valve FF has flow rate=0; tunnels lead to valves EE, GG
Valve GG has flow rate=0; tunnels lead to valves FF, HH
Valve HH has flow rate=22; tunnel leads to valve GG
Valve II has flow rate=0; tunnels lead to valves AA, JJ
Valve JJ has flow rate=21; tunnel leads to valve II
""",
]


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUTS[0]) as f:
        yield f


def test_parse_valves(sample_input):
    result = parse_valves(sample_input)
    assert len(result.nodes) == 10
    assert len(result.edges) == 10


@pytest.mark.parametrize(("with_elephant", "expected"), [(False, 1651), (True, 1707)])
def test_find_max_release(sample_input, with_elephant, expected):
    graph = parse_valves(sample_input)
    if with_elephant:
        nav = Navigator2(graph)
        initial_state = State2((Valve("AA", 0), Valve("AA", 0)))
    else:
        nav = Navigator(graph)
        initial_state = State(Valve("AA", 0))
    assert find_max_release(nav, initial_state) == expected
