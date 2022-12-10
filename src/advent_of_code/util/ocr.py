import atexit
import math
from io import StringIO
from typing import ContextManager, Iterator, TextIO

import yaml
from pkg_resources import cleanup_resources, resource_filename

DEFAULT_BLOCK = "▒"


def to_string(blocks: list[str], block_char: str = DEFAULT_BLOCK) -> str:
    char_count = math.ceil(len(blocks[0]) / 5)
    refs = load_references()

    def generate_chars() -> Iterator[str]:
        for i in range(char_count):
            block = tuple(
                line[5 * i : 5 * i + 4].rstrip().replace(block_char, DEFAULT_BLOCK)
                for line in blocks
            )
            yield refs[block]

    return "".join(generate_chars())


def load_references() -> dict[tuple[str, ...], str]:
    with open(resource_filename("advent_of_code", "resources/ocr.yaml")) as f:
        letters_to_blocks = yaml.safe_load(f)["characters"]
    atexit.register(cleanup_resources)
    return {
        tuple(line.rstrip() for line in block): char for char, block in letters_to_blocks.items()
    }


class PrintToString(ContextManager[TextIO]):
    def __init__(self, block_char: str = DEFAULT_BLOCK):
        self._buffer = StringIO()
        self._block_char = block_char

    def __enter__(self) -> TextIO:
        return self._buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    def to_string(self) -> str:
        self._buffer.seek(0)
        blocks = [line.rstrip() for line in self._buffer.readlines()]
        return to_string(blocks, self._block_char)
