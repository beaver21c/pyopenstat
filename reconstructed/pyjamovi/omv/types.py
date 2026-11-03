"""
pyjamovi Type Definitions
Matches jamovi internal types and constants.
"""

from enum import Enum


class ColumnType(str, Enum):
    DATA = "Data"
    COMPUTED = "Computed"
    TRANSFORMED = "Transformed"
    FILTER = "Filter"
    OUTPUT = "Output"

    @classmethod
    def parse(cls, value: str):
        for member in cls:
            if member.value.lower() == str(value).lower():
                return member
        return cls.DATA


class DataType(str, Enum):
    DECIMAL = "Decimal"
    INTEGER = "Integer"
    TEXT = "Text"

    @classmethod
    def parse(cls, value: str):
        for member in cls:
            if member.value.lower() == str(value).lower():
                return member
        return cls.DECIMAL


class MeasureType(str, Enum):
    CONTINUOUS = "Continuous"
    NOMINAL = "Nominal"
    ORDINAL = "Ordinal"
    ID = "ID"

    @classmethod
    def parse(cls, value: str):
        for member in cls:
            if member.value.lower() == str(value).lower():
                return member
        return cls.NOMINAL


INT_MISSING = -2147483648
FLOAT_MISSING = float("nan")
