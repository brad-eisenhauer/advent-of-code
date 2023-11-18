"""Advent of Code 2016, day 10: https://adventofcode.com/2016/day/10"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import IO, Callable, ClassVar, Iterator, NewType, Optional

from advent_of_code.base import Solution
from advent_of_code.cli import log


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(10, 2016, **kwargs)

    def solve_part_one(self) -> int:
        env = WorkingEnvironment()
        with self.open_input() as fp:
            for event in env.run(fp):
                if event.type == "compare" and event.values == [17, 61]:
                    return event.source.id

    def solve_part_two(self) -> int:
        env = WorkingEnvironment()
        with self.open_input() as fp:
            for _ in env.run(fp):
                pass
        return env.outputs[0].contents[0] * env.outputs[1].contents[0] * env.outputs[2].contents[0]


Microchip = NewType("Microchip", int)


@dataclass
class Event:
    source: "Robot"
    type: str
    values: list[int]


@dataclass
class EventSubscription:
    callback: Callable[[Event], None]
    event_type: Optional[str] = None
    id: int = field(default_factory=lambda: EventSubscription.next_id)
    next_id: ClassVar[int] = 0

    def __post_init__(self):
        EventSubscription.next_id += 1


@dataclass
class WorkingEnvironment:
    robots: dict[int, "Robot"] = field(default_factory=dict)
    outputs: dict[int, "Output"] = field(default_factory=dict)
    subscriptions: list[EventSubscription] = field(default_factory=list)
    event_queue: deque[Event] = field(default_factory=deque)

    def publish_event(self, event: Event):
        for subscription in self.subscriptions:
            if subscription.event_type is None or subscription.event_type == event.type:
                subscription.callback(event)

    def subscribe(self, callback: Callable[[Event], None], event_type: Optional[str] = None) -> int:
        subscription = EventSubscription(callback, event_type)
        self.subscriptions.append(subscription)
        return subscription.id

    def unsubscribe(self, subscription_id: int):
        self.subscriptions = [s for s in self.subscriptions if s.id != subscription_id]

    def run(self, instructions: IO) -> Iterator[Event]:
        sources: list[str] = []
        bot_pattern = r"bot (\d+)"
        output_pattern = r"output (\d+)"
        for line in instructions:
            if (match := re.match(bot_pattern, line)) is None:
                sources.append(line)
            else:
                robot_id = int(match.group(1))
                self.robots[robot_id] = Robot(robot_id, line.strip(), self)
            for match in re.findall(output_pattern, line):
                output_id = int(match)
                if output_id not in self.outputs:
                    self.outputs[output_id] = Output()

        self.subscribe(self.enqueue_event)

        for line in sources:
            initial_pattern = r"value (?P<value>\d+) goes to bot (?P<dest>\d+)"
            match = re.match(initial_pattern, line)
            value = int(match.groupdict()["value"])
            bot_id = int(match.groupdict()["dest"])
            self.robots[bot_id].receive(Microchip(value))

        while self.event_queue:
            event = self.event_queue.popleft()
            yield event
            match event.type:
                case "received":
                    event.source.execute_instruction()

    def enqueue_event(self, event: Event):
        self.event_queue.append(event)


@dataclass
class Robot:
    id: int
    instruction: str
    environment: WorkingEnvironment
    holding: list[Microchip] = field(default_factory=list)

    def receive(self, chip: Microchip):
        self.holding.append(chip)
        self.environment.publish_event(Event(self, "received", [chip]))

    def execute_instruction(self):
        if len(self.holding) < 2:
            return
        pattern = r"gives low to (?P<low_dest>(?:bot|output) \d+) and high to (?P<high_dest>(?:bot|output) \d+)"
        match = re.search(pattern, self.instruction)
        low_chip, high_chip = sorted(self.holding[:2])
        self.environment.publish_event(Event(self, "compare", [low_chip, high_chip]))
        self.holding = self.holding[2:]
        self.send_chip(low_chip, match.groupdict()["low_dest"])
        self.send_chip(high_chip, match.groupdict()["high_dest"])

    def send_chip(self, chip: Microchip, dest: str):
        dest_type, dest_id = dest.split()
        dest_id = int(dest_id)
        match dest_type:
            case "bot":
                self.environment.robots[dest_id].receive(chip)
            case "output":
                self.environment.outputs[dest_id].receive(chip)


@dataclass
class Output:
    contents: list[Microchip] = field(default_factory=list)

    def receive(self, chip: Microchip):
        self.contents.append(chip)
