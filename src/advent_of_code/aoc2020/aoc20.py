"""Advent of Code 2020, Day 20: https://adventofcode.com/2020/day/20"""
from __future__ import annotations

import logging
import operator
from enum import Enum, auto
from functools import cached_property, reduce
from io import StringIO
from itertools import chain, product, takewhile
from math import sqrt
from typing import Iterable, Iterator, Mapping, Optional, Sequence, TextIO, Tuple

from advent_of_code.base import Solution
from advent_of_code.util import make_sequence

log = logging.getLogger("aoc")

SEA_MONSTER = """
                  # 
#    ##    ##    ###
 #  #  #  #  #  #   
"""


class AocSolution(Solution[int]):
    def __init__(self, **kwargs):
        super().__init__(20, 2020, **kwargs)
        self._image: Optional[Image] = None

    @property
    def image(self) -> Optional[Image]:
        if self._image is None:
            log.debug("Assembling image...")
            with self.open_input() as f:
                tiles = read_tiles(f)
                self._image = assemble_image(tiles)
        return self._image

    def solve_part_one(self) -> int:
        corner_product = reduce(
            operator.mul, (self.image[row][col].id for row, col in product((0, -1), (0, -1)))
        )
        return corner_product

    def solve_part_two(self) -> int:
        log.debug("Searching for sea monsters...")
        monster_mask = get_monster_mask()
        monsters = make_sequence(find_sea_monsters(self.image, monster_mask))
        log.debug("Found %d sea monsters.", len(monsters))

        return calc_water_roughness(self.image, monsters)


class Tile:
    def __init__(self, id: int, img_data: Iterable[str]):
        self.id = id
        self.img_data = tuple(s.strip() for s in img_data)
        self.size = len(self.img_data)
        if self.size == 0:
            raise ValueError("Empty tile data.")
        if any(len(s) != self.size for s in self.img_data):
            raise ValueError("Image data too short.")

    @cached_property
    def edges(self) -> Mapping[Edge, str]:
        return {
            Edge.TOP: self.img_data[0],
            Edge.BOTTOM: self.img_data[-1],
            Edge.LEFT: "".join(s[0] for s in self.img_data),
            Edge.RIGHT: "".join(s[-1] for s in self.img_data),
        }

    def rotate_left(self) -> Tile:
        new_data = ("".join(s[i - 1] for s in self.img_data) for i in range(self.size, 0, -1))
        return Tile(self.id, new_data)

    def rotate_right(self) -> Tile:
        new_data = ("".join(s[i] for s in reversed(self.img_data)) for i in range(self.size))
        return Tile(self.id, new_data)

    def flip_horizontal(self) -> Tile:
        new_data = (s[::-1] for s in self.img_data)
        return Tile(self.id, new_data)

    def flip_vertical(self) -> Tile:
        new_data = self.img_data[::-1]
        return Tile(self.id, new_data)

    def find_matches(self, edge_requirements: Mapping[Edge, str]) -> Iterator[Tile]:
        """Find transformations of the current Tile that satisfy the given edges"""
        for tile in self.generate_transformations():
            tile_edges = tile.edges
            if all(tile_edges[edge] == e for edge, e in edge_requirements.items()):
                yield tile

    def generate_transformations(self) -> Iterator[Tile]:
        """Generate reflections and rotations of the Tile"""
        next_variation = self
        for _ in range(4):
            yield next_variation
            yield next_variation.flip_vertical()
            next_variation = next_variation.rotate_right()

    def strip_edges(self) -> Iterator[str]:
        return (s[1:-1] for s in self.img_data[1:-1])


def read_tiles(fp: TextIO) -> Iterator[Tile]:
    try:
        while True:
            id_line = fp.readline()
            tile_id = int(id_line.split()[1][:-1])
            tile_data = takewhile(lambda s: s.strip() != "", fp)
            yield Tile(tile_id, tile_data)
    except (IOError, ValueError, IndexError) as e:
        ...


class Image:
    """Two-dimensional array of tiles"""

    def __init__(self, tiles: Iterable[Iterable[Optional[Tile]]]):
        self.tiles = tuple(tuple(row) for row in tiles)

    def __getitem__(self, index):
        return self.tiles[index]

    def __len__(self):
        return len(self.tiles)

    def merge(self) -> Iterator[str]:
        """Strip tile edges and merge into a single tile"""
        if not self.is_complete:
            raise ValueError("Cannot merge an incomplete image.")
        for tile_row in self.tiles:
            for tile_lines in zip(*(tile.strip_edges() for tile in tile_row)):
                yield "".join(tile_lines)

    def get(self, row: int, col: int) -> Optional[Tile]:
        try:
            return self[row][col]
        except IndexError:
            return None

    def insert(self, tile: Tile, row: int, col: int) -> Image:
        row_range = range(min(0, row), len(self.tiles))
        col_range = range(min(0, col), len(self.tiles[0]))
        new_data = (
            (
                tile if row_idx == row and col_idx == col else self.get(row_idx, col_idx)
                for col_idx in col_range
            )
            for row_idx in row_range
        )
        return Image(new_data)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return len(self.tiles), len(self.tiles[0])

    @property
    def is_complete(self) -> bool:
        return all(all(tile is not None for tile in row) for row in self)

    @property
    def is_empty(self) -> bool:
        return all(all(tile is None for tile in row) for row in self)

    @classmethod
    def new(cls, size: int) -> Image:
        return cls(((None,) * size,) * size)

    def locations(self) -> Iterator[Tuple[int, int]]:
        return product(range(len(self.tiles)), range(len(self.tiles[0])))

    def calc_edge_requirements(self, row: int, col: int) -> Mapping[Edge, str]:
        """Calculate the required hashes for each edge bordering an existing Tile"""
        adjacent_offsets = {
            Edge.TOP: (-1, 0),
            Edge.BOTTOM: (1, 0),
            Edge.LEFT: (0, -1),
            Edge.RIGHT: (0, 1),
        }
        edge_requirements = {}
        for edge in Edge:
            if (
                adj_tile := self.get(
                    *(idx + offset for idx, offset in zip((row, col), adjacent_offsets[edge]))
                )
            ) is not None:
                edge_requirements[edge] = adj_tile.edges[edge.opposite]
        return edge_requirements


def assemble_image(tiles: Iterable[Tile]) -> Optional[Image]:
    tiles = make_sequence(tiles)
    if not tiles:
        return None

    size = int(sqrt(len(tiles)))
    image = Image.new(size)

    for tile in find_corner_tiles(tiles):
        result = _assemble_image(
            (t for t in tiles if t is not tile), image.insert(tile, 0, 0), 0, 1
        )
        if result is not None:
            return result

    raise ValueError("Unable to solve image.")


def _assemble_image(
    tiles: Iterable[Tile], image: Image, row_idx: int, col_idx: int
) -> Optional[Image]:
    tiles = make_sequence(tiles)

    if not tiles:
        return image if image.is_complete else None

    next_col = (col_idx + 1) % image.dimensions[1]
    next_row = row_idx + int(next_col < col_idx)

    edge_requirements = image.calc_edge_requirements(row_idx, col_idx)
    for tile in tiles:
        for candidate_match in tile.find_matches(edge_requirements):
            result = _assemble_image(
                (t for t in tiles if t is not tile),
                image.insert(candidate_match, row_idx, col_idx),
                next_row,
                next_col,
            )
            if result is not None:
                return result

    return None


def find_corner_tiles(tiles: Sequence[Tile]) -> Iterator[Tile]:
    """
    Corner tile is any tile with exactly two adjacent edges whose hashes cannot be
    matched by any other tile.  Tile will be rotated or flipped so unmatched edges
    are top and left.
    """
    for tile in tiles:
        unmatched_edges = []
        for edge in Edge:
            edge_requirements = {edge.opposite: tile.edges[edge]}
            for adj_tile in tiles:
                if (
                    adj_tile is not tile
                    and next(adj_tile.find_matches(edge_requirements), None) is not None
                ):
                    break
            else:
                unmatched_edges.append(edge)
        if len(unmatched_edges) == 2 and Edge.are_adjacent(*unmatched_edges):
            if Edge.TOP not in unmatched_edges:
                tile = tile.flip_vertical()
            if Edge.LEFT not in unmatched_edges:
                tile = tile.flip_horizontal()
            yield tile


def get_monster_mask():
    """It was a sea-floor smash."""
    with StringIO(SEA_MONSTER) as sm:
        return Mask(
            (x, y) for x, line in enumerate(sm) for y, char in enumerate(line) if char == "#"
        )


class Mask:
    def __init__(self, points: Iterable[Tuple[int, int]]):
        self.points: set[Tuple[int, int]] = set(points)

    @cached_property
    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        min_x = min(x for x, _ in self.points)
        max_x = max(x for x, _ in self.points)
        min_y = min(y for _, y in self.points)
        max_y = max(y for _, y in self.points)
        return (min_x, min_y), (max_x, max_y)

    @cached_property
    def dimensions(self) -> Tuple[int, int]:
        (min_x, min_y), (max_x, max_y) = self.bounds
        return max_x - min_x + 1, max_y - min_y + 1

    def normalize(self) -> Mask:
        (x_offset, y_offset), _ = self.bounds
        return self.translate(-x_offset, -y_offset)

    def rotate_left(self) -> Mask:
        return Mask((-x, y) for x, y in self.transpose())

    def rotate_right(self) -> Mask:
        return Mask((x, -y) for x, y in self.transpose())

    def flip_horizontal(self) -> Mask:
        return Mask((-x, y) for x, y in self.points)

    def flip_vertical(self) -> Mask:
        return Mask((x, -y) for x, y in self.points)

    def translate(self, x_offset: int, y_offset: int) -> Mask:
        return Mask((x + x_offset, y + y_offset) for x, y in self.points)

    def transpose(self) -> Iterator[Tuple[int, int]]:
        for x, y in self.points:
            yield y, x

    def generate_transformations(self) -> Iterator[Mask]:
        """Generate rotations and reflections of the Mask"""
        next_variation = self
        for _ in range(4):
            yield next_variation.normalize()
            yield next_variation.flip_vertical().normalize()
            next_variation = next_variation.rotate_right()


def find_sea_monsters(image: Image, monster_mask: Mask) -> Iterator[Mask]:
    image_data = make_sequence(image.merge())
    image_dims = len(image_data), len(image_data[0])
    for mask_variation in monster_mask.generate_transformations():
        mask_dims = mask_variation.dimensions
        x_range = range(image_dims[0] - mask_dims[0] + 1)
        y_range = range(image_dims[1] - mask_dims[1] + 1)
        for x_offset, y_offset in product(x_range, y_range):
            candidate_mask = mask_variation.translate(x_offset, y_offset)
            if does_mask_match(image_data, candidate_mask):
                yield candidate_mask


def does_mask_match(image_data: Sequence[str], mask: Mask) -> bool:
    return all(image_data[x][y] == "#" for x, y in mask.points)


def calc_water_roughness(image: Image, monsters: Iterable[Mask]) -> int:
    monster_points = set(chain.from_iterable(m.points for m in monsters))
    result = sum(
        1
        for x, row in enumerate(image.merge())
        for y, char in enumerate(row)
        if char == "#" and (x, y) not in monster_points
    )
    return result


TEST_INPUT = """Tile 2311:
..##.#..#.
##..#.....
#...##..#.
####.#...#
##.##.###.
##...#.###
.#.#.#..##
..#....#..
###...#.#.
..###..###

Tile 1951:
#.##...##.
#.####...#
.....#..##
#...######
.##.#....#
.###.#####
###.##.##.
.###....#.
..#.#..#.#
#...##.#..

Tile 1171:
####...##.
#..##.#..#
##.#..#.#.
.###.####.
..###.####
.##....##.
.#...####.
#.##.####.
####..#...
.....##...

Tile 1427:
###.##.#..
.#..#.##..
.#.##.#..#
#.#.#.##.#
....#...##
...##..##.
...#.#####
.#.####.#.
..#..###.#
..##.#..#.

Tile 1489:
##.#.#....
..##...#..
.##..##...
..#...#...
#####...#.
#..#.#.#.#
...#.#.#..
##.#...##.
..##.##.##
###.##.#..

Tile 2473:
#....####.
#..#.##...
#.##..#...
######.#.#
.#...#.#.#
.#########
.###.#..#.
########.#
##...##.#.
..###.#.#.

Tile 2971:
..#.#....#
#...###...
#.#.###...
##.##..#..
.#####..##
.#..####.#
#..#.#..#.
..####.###
..#.#.###.
...#.#.#.#

Tile 2729:
...#.#.#.#
####.#....
..#.#.....
....#..#.#
.##..##.#.
.#.####...
####.#.#..
##.####...
##..#.##..
#.##...##.

Tile 3079:
#.#.#####.
.#..######
..#.......
######....
####.#..#.
.#...#.##.
#.#####.##
..#.###...
..#.......
..#.###...
"""


class Edge(Enum):
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()

    @staticmethod
    def are_adjacent(a: Edge, b: Edge) -> bool:
        return not Edge.are_opposite(a, b)

    @staticmethod
    def are_opposite(a: Edge, b: Edge) -> bool:
        return a.opposite is b

    @cached_property
    def opposite(self):
        return {
            Edge.TOP: Edge.BOTTOM,
            Edge.BOTTOM: Edge.TOP,
            Edge.LEFT: Edge.RIGHT,
            Edge.RIGHT: Edge.LEFT,
        }[self]
