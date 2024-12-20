import logging
from abc import ABC, abstractmethod
from collections import deque
from heapq import heappop, heappush
from typing import Generic, Iterator, Optional, TypeVar

import networkx as nx

log = logging.getLogger("aoc")

N = TypeVar("N")
T = TypeVar("T")
S = TypeVar("S")


class AStar(Generic[S], ABC):
    """
    A* implementation based on https://www.redblobgames.com/pathfinding/a-star/introduction.html
    """

    class Queue:
        """Priority queue"""

        def __init__(self):
            self._contents: list[tuple[int, S]] = []

        def __bool__(self):
            return bool(self._contents)

        def __len__(self):
            return len(self._contents)

        def peek(self) -> S:
            return self._contents[0]

        def pop(self) -> S:
            _, result = heappop(self._contents)
            return result

        def push(self, priority: int, item: S):
            heappush(self._contents, (priority, item))

    @abstractmethod
    def is_goal_state(self, state: S) -> bool:
        ...

    @abstractmethod
    def generate_next_states(self, state: S) -> Iterator[tuple[int, S]]:
        """For a given state, generate successor states and the transition cost."""
        ...

    def heuristic(self, state: S) -> int:
        """With no heuristic, A* is equivalent to Djikstra."""
        return 0

    def _find_min_cost_path(
        self, initial_state: S
    ) -> tuple[S, dict[S, int], dict[S, Optional[S]]]:
        accumulated_cost: dict[S, int] = {initial_state: 0}
        came_from: dict[S, Optional[S]] = {initial_state: None}
        frontier = self.Queue()
        frontier.push(0, initial_state)
        state_count = 0
        while frontier:
            current_state = frontier.pop()
            state_count += 1
            if self.is_goal_state(current_state):
                log.debug("Evaluated %d states.", state_count)
                return current_state, accumulated_cost, came_from
            current_cost = accumulated_cost[current_state]
            for cost, next_state in self.generate_next_states(current_state):
                total_cost = current_cost + cost
                if next_state not in accumulated_cost or total_cost < accumulated_cost[next_state]:
                    accumulated_cost[next_state] = total_cost
                    came_from[next_state] = current_state
                    frontier.push(total_cost + self.heuristic(next_state), next_state)
        raise ValueError("Goal state not found.")

    def find_min_cost_path(self, initial_state: S) -> Iterator[tuple[S, int]]:
        goal_state, accumulated_cost, came_from = self._find_min_cost_path(initial_state)
        backtrack = []
        state = goal_state
        while state is not None:
            backtrack.append((state, accumulated_cost[state]))
            state = came_from[state]
        return reversed(backtrack)

    def find_min_cost_to_goal(self, initial_state: S) -> int:
        goal_state, accumulated_cost, _ = self._find_min_cost_path(initial_state)
        return accumulated_cost[goal_state]


class BFS(AStar[S], ABC):
    class Queue:
        """Simple queue"""

        def __init__(self):
            self._contents: deque[S] = deque()

        def __bool__(self):
            return bool(self._contents)

        def __len__(self):
            return len(self._contents)

        def peek(self) -> S:
            return self._contents[0]

        def pop(self) -> S:
            return self._contents.popleft()

        def push(self, _priority: int, item: S):
            self._contents.append(item)


class GraphSimplifier(Generic[N]):
    DEAD_END = 1
    HALLWAY = 2

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @abstractmethod
    def is_protected(self, node: N, mode: int) -> bool:
        ...

    def simplify(self, depth: int = 1):
        g = self.graph
        nodes_to_remove = []
        for node in g.nodes:
            neighbors = list(g.neighbors(node))
            match len(neighbors):
                case 1:  # dead end
                    if self.is_protected(node, self.DEAD_END):
                        continue
                    nodes_to_remove.append(node)
                    g.remove_edge(node, neighbors[0])
                case 2:  # hallway
                    if self.is_protected(node, self.HALLWAY):
                        continue
                    nodes_to_remove.append(node)
                    weight = sum(g.get_edge_data(node, n)["weight"] for n in neighbors)
                    for n in neighbors:
                        g.remove_edge(node, n)
                    g.add_edge(*neighbors, weight=weight)
                case _:
                    ...
        if nodes_to_remove:
            g.remove_nodes_from(nodes_to_remove)
            self.simplify(depth + 1)
        else:
            log.debug("Ran simplify %d times.", depth)
            log.debug("Simplified graph has %d nodes.", len(g.nodes))
