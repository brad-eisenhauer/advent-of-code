"""Advent of Code 2023, day 5: https://adventofcode.com/2023/day/5"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import reduce
from io import StringIO
from itertools import chain
from typing import IO, Iterator, Sequence

import pytest

from advent_of_code.base import Solution
from advent_of_code.cli import log
from advent_of_code.util import create_groups


class AocSolution(Solution[int, int]):
    def __init__(self, **kwargs):
        super().__init__(5, 2023, **kwargs)

    def solve_part_one(self) -> int:
        log.debug("Reading recipe...")
        with self.open_input() as fp:
            seeds, maps = read_recipe(fp)
        log.debug("Seeds=%s", seeds)
        log.debug("Maps=%s", maps)
        locations = []
        for seed_id in seeds:
            loc = find_location_for_seed(seed_id, maps)
            log.debug("Seed %d => Location %d", seed_id, loc)
            locations.append(loc)
        return min(locations)

    def solve_part_two(self) -> int:
        with self.open_input() as fp:
            seeds, maps = read_recipe(fp)
        log.debug("seeds=%s", seeds)
        composed_map = reduce(lambda a, b: b.compose(a), maps)
        map_thresholds = set(
            chain.from_iterable(
                (rng.source, rng.source + rng.length) for rng in composed_map.ranges
            )
        )
        log.debug("thresholds=%s", map_thresholds)
        seed_thresholds = set(seeds[0::2])
        return min(
            composed_map(n)
            for n in map_thresholds | seed_thresholds
            if any(lo <= n < lo + length for lo, length in create_groups(seeds, 2))
        )


@dataclass(frozen=True)
class Range:
    dest: int
    source: int
    length: int

    def __iter__(self):
        return iter((self.dest, self.source, self.length))

    def __sub__(self, other: Range) -> Iterator[Range]:
        if self.source < other.source:
            yield Range(self.dest, self.source, min(self.length, other.source - self.source))
        self_source_end = self.source + self.length
        other_source_end = other.source + other.length
        if other_source_end < self_source_end:
            length = min(self.length, self_source_end - other_source_end)
            source = self_source_end - length
            dest = self.dest + (source - self.source)
            yield Range(dest, source, length)


@dataclass
class RangeMap:
    ranges: set[Range]

    @classmethod
    def read(cls, fp: IO) -> RangeMap:
        ranges: set[Range] = set()
        while line := fp.readline().strip():
            ranges.add(Range(*(int(n) for n in line.split())))
        return cls(ranges)

    def __call__(self, n: int) -> int:
        for rng in self.ranges:
            if rng.source <= n < rng.source + rng.length:
                return rng.dest + (n - rng.source)
        return n

    def compose(self, other: RangeMap) -> RangeMap:
        bare_ranges = list(
            chain.from_iterable(
                reduce(
                    lambda rngs, rng: list(chain.from_iterable((r - rng) for r in rngs)),
                    other.ranges,
                    [rng],
                )
                for rng in self.ranges
            )
        )
        intersected_ranges: list[Range] = []
        rng_queue = deque(other.ranges)
        while rng_queue:
            pre_rng = rng_queue.popleft()
            for post_rng in self.ranges:
                if pre_rng.dest <= post_rng.source < pre_rng.dest + pre_rng.length:
                    rng_queue.extend(
                        pre_rng
                        - Range(0, pre_rng.source + post_rng.source - pre_rng.dest, post_rng.length)
                    )
                    length = min(post_rng.length, pre_rng.dest + pre_rng.length - post_rng.source)
                    dest = post_rng.dest
                    source = pre_rng.source + post_rng.source - pre_rng.dest
                    intersected_ranges.append(Range(dest, source, length))
                    break
                if post_rng.source <= pre_rng.dest < post_rng.source + post_rng.length:
                    rng_queue.extend(
                        pre_rng
                        - Range(0, pre_rng.source, post_rng.source + post_rng.length - pre_rng.dest)
                    )
                    length = min(pre_rng.length, post_rng.source + post_rng.length - pre_rng.dest)
                    source = pre_rng.source
                    dest = post_rng.dest + pre_rng.dest - post_rng.source
                    intersected_ranges.append(Range(dest, source, length))
                    break
            else:
                intersected_ranges.append(pre_rng)
        return RangeMap(set(bare_ranges + intersected_ranges))


def read_recipe(fp: IO) -> tuple[set[int], Sequence[RangeMap]]:
    seed_line = fp.readline().strip()
    _, seed_ids_str = seed_line.split(": ")
    seed_ids = [int(n) for n in seed_ids_str.split()]
    fp.readline()

    maps: list[RangeMap] = []
    while _ := fp.readline().strip():
        maps.append(RangeMap.read(fp))

    return seed_ids, maps


def find_location_for_seed(seed_id: int, maps: Sequence[RangeMap]) -> int:
    result = seed_id
    for map in maps:
        result = map(result)
    return result


SAMPLE_INPUTS = [
    """\
seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4
""",
]


@pytest.fixture
def sample_input(request):
    with StringIO(SAMPLE_INPUTS[getattr(request, "param", 0)]) as f:
        yield f


def test_read_recipe(sample_input):
    seeds, maps = read_recipe(sample_input)
    assert seeds == [79, 14, 55, 13]
    assert maps == [
        RangeMap({Range(50, 98, 2), Range(52, 50, 48)}),
        RangeMap({Range(0, 15, 37), Range(37, 52, 2), Range(39, 0, 15)}),
        RangeMap({Range(49, 53, 8), Range(0, 11, 42), Range(42, 0, 7), Range(57, 7, 4)}),
        RangeMap({Range(88, 18, 7), Range(18, 25, 70)}),
        RangeMap({Range(45, 77, 23), Range(81, 45, 19), Range(68, 64, 13)}),
        RangeMap({Range(0, 69, 1), Range(1, 0, 69)}),
        RangeMap({Range(60, 56, 37), Range(56, 93, 4)}),
    ]


def test_find_location_for_seed_single(sample_input):
    seeds, maps = read_recipe(sample_input)
    result = [find_location_for_seed(seed_id, [maps[0]]) for seed_id in seeds]
    assert result == [81, 14, 57, 13]


def test_find_location_for_seed_all(sample_input):
    seeds, maps = read_recipe(sample_input)
    result = [find_location_for_seed(seed_id, maps) for seed_id in seeds]
    assert result == [82, 43, 86, 35]


def test_compose_range_maps():
    rm1 = RangeMap({Range(50, 98, 2), Range(52, 50, 48)})
    rm2 = RangeMap({Range(0, 15, 37), Range(37, 52, 2), Range(39, 0, 15)})
    expected = RangeMap(
        {
            Range(39, 0, 15),
            Range(0, 15, 35),
            Range(37, 50, 2),
            Range(54, 52, 46),
            Range(35, 98, 2),
        }
    )
    for n in range(100):
        assert rm2(rm1(n)) == expected(n)
    assert rm2.compose(rm1) == expected


class TestRange:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (Range(5, 0, 10), Range(5, 5, 10), [Range(5, 0, 5)]),
            (Range(10, 5, 10), Range(5, 0, 10), [Range(15, 10, 5)]),
            (Range(5, 10, 15), Range(0, 15, 5), [Range(5, 10, 5), Range(15, 20, 5)]),
            (Range(5, 0, 10), Range(5, 5, 0), [Range(5, 0, 5), Range(10, 5, 5)]),
            (Range(0, 5, 5), Range(5, 0, 10), []),
        ],
    )
    def test_subtract(self, left, right, expected):
        assert list(left - right) == expected
