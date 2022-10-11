"""Advent of Code 2020, day 4: https://adventofcode.com/2020/day/4"""
import re
from io import StringIO
from typing import Any, Iterator, NewType, Protocol, Sequence, TextIO

import pytest

from advent_of_code.base import Solution

Passport = NewType("Passport", dict[str, Any])


class AocSolution(Solution[int]):
    def __init__(self):
        super().__init__(4, 2020)

    def solve_part_one(self) -> int:
        with self.open_input() as f:
            return sum(1 for p in parse_passports(f) if validate_passport(p))

    def solve_part_two(self) -> int:
        with self.open_input() as f:
            return sum(1 for p in parse_passports(f) if validate_passport(p, strict=True))


class PassportValidator:
    KEY = ""

    @classmethod
    def validate(cls, passport: Passport, strict: bool = False) -> bool:
        if not cls.validate_key(passport):
            return False
        if not strict:
            return True
        return cls._validate(passport)

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        ...

    @classmethod
    def validate_key(cls, passport: Passport) -> bool:
        return cls.KEY in passport

    @classmethod
    def validate_year(cls, passport: Passport, year_range: range) -> bool:
        if cls.KEY not in passport:
            return False
        if len(passport[cls.KEY]) != 4:
            return False
        try:
            year = int(passport[cls.KEY])
            if year not in year_range:
                return False
        except ValueError:
            return False
        return True


class BirthYearValidator(PassportValidator):
    KEY = "byr"

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        if cls.KEY not in passport:
            return False
        return cls.validate_year(passport, range(1920, 2003))


class IssueYearValidator(PassportValidator):
    KEY = "iyr"

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        return cls.validate_year(passport, range(2010, 2021))


class ExpirationYearValidator(PassportValidator):
    KEY = "eyr"

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        return cls.validate_year(passport, range(2020, 2031))


class HeightValidator(PassportValidator):
    KEY = "hgt"
    PATTERN = re.compile(r"(\d{2,3})(in|cm)$")

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        re_match = cls.PATTERN.match(passport[cls.KEY])
        if not re_match:
            return False
        height = int(re_match.groups()[0])
        match re_match.groups()[1]:
            case "in":
                if height not in range(59, 77):
                    return False
            case "cm":
                if height not in range(150, 194):
                    return False
            case _:
                return False
        return True


class HairColorValidator(PassportValidator):
    KEY = "hcl"
    PATTERN = re.compile(r"^#[0-9a-f]{6}$")

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        if cls.PATTERN.match(passport[cls.KEY]) is None:
            return False
        return True


class EyeColorValidator(PassportValidator):
    KEY = "ecl"

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        if passport[cls.KEY] not in ("amb", "blu", "brn", "gry", "grn", "hzl", "oth"):
            return False
        return True


class PassportIDValidator(PassportValidator):
    KEY = "pid"

    @classmethod
    def _validate(cls, passport: Passport) -> bool:
        passport_id = passport[cls.KEY]
        if len(passport_id) != 9:
            return False
        if not passport_id.isnumeric():
            return False
        return True


VALIDATORS = (
    BirthYearValidator,
    IssueYearValidator,
    ExpirationYearValidator,
    HeightValidator,
    HairColorValidator,
    EyeColorValidator,
    PassportIDValidator,
)


def parse_passports(f: TextIO) -> Iterator[Passport]:
    for passport_text in f.read().split("\n\n"):
        next_pp = {}
        for field in passport_text.split():
            key, value = field.split(":")
            next_pp[key] = value
        yield next_pp


def validate_passport(
    passport: Passport, validators: Sequence[PassportValidator] = VALIDATORS, strict: bool = False
) -> bool:
    return all(validator.validate(passport, strict) for validator in validators)


SAMPLE_INPUT = """\
ecl:gry pid:860033327 eyr:2020 hcl:#fffffd
byr:1937 iyr:2017 cid:147 hgt:183cm

iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884
hcl:#cfa07d byr:1929

hcl:#ae17e1 iyr:2013
eyr:2024
ecl:brn pid:760753108 byr:1931
hgt:179cm

hcl:#cfa07d eyr:2025 pid:166559648
iyr:2011 ecl:brn hgt:59in
"""


@pytest.fixture
def sample_input():
    with StringIO(SAMPLE_INPUT) as f:
        yield f


def test_parse_passports(sample_input):
    pp = list(parse_passports(sample_input))
    expected = [
        {
            "ecl": "gry",
            "pid": "860033327",
            "eyr": "2020",
            "hcl": "#fffffd",
            "byr": "1937",
            "iyr": "2017",
            "cid": "147",
            "hgt": "183cm",
        },
        {
            "iyr": "2013",
            "ecl": "amb",
            "cid": "350",
            "eyr": "2023",
            "pid": "028048884",
            "hcl": "#cfa07d",
            "byr": "1929",
        },
        {
            "hcl": "#ae17e1",
            "iyr": "2013",
            "eyr": "2024",
            "ecl": "brn",
            "pid": "760753108",
            "byr": "1931",
            "hgt": "179cm",
        },
        {
            "hcl": "#cfa07d",
            "eyr": "2025",
            "pid": "166559648",
            "iyr": "2011",
            "ecl": "brn",
            "hgt": "59in",
        },
    ]
    assert pp == expected


def test_validate_passport(sample_input):
    results = [validate_passport(p) for p in parse_passports(sample_input)]
    expected = [True, False, True, False]
    assert results == expected
