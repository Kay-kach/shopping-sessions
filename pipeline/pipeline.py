#!/usr/bin/env python3
"""Shopping Sessions reference-correctness implementation using Polars output.

Reads DATA_DIR and writes user_daily/ and campaign_daily/ Parquet datasets.
"""
import bisect
import glob
import json
import os
import re
from datetime import date, datetime, timedelta, timezone

NCPU = os.environ.get("NCPU")
if NCPU:
    os.environ.setdefault("POLARS_MAX_THREADS", NCPU)

import polars as pl  # noqa: E402

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")

MS_PER_DAY = 86_400_000
MS_PER_7_DAYS = 7 * MS_PER_DAY
SESSION_GAP_MS = 1_800_000
INT64_MAX = 9_223_372_036_854_775_807
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

RE_F1 = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z$"
)
RE_F2 = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})([+-])([0-9]{2}):([0-9]{2})$"
)
RE_DIGITS = re.compile(r"^[0-9]+$")


def _datetime_to_ms(dt):
    """Convert an aware UTC datetime to epoch milliseconds without floats."""
    delta = dt - EPOCH
    return delta.days * MS_PER_DAY + delta.seconds * 1000 + delta.microseconds // 1000


def parse_ts(value):
    """Parse the three timestamp forms specified by the challenge."""
    if value == "":
        return None

    match = RE_F1.fullmatch(value)
    if match:
        try:
            y, mo, d, h, mi, sec = (int(part) for part in match.groups())
            dt = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc)
            return _datetime_to_ms(dt)
        except (ValueError, OverflowError):
            return None

    match = RE_F2.fullmatch(value)
    if match:
        try:
            y, mo, d, h, mi, sec = (int(part) for part in match.groups()[:6])
            sign = match.group(7)
            offset_hours = int(match.group(8))
            offset_minutes = int(match.group(9))
            if offset_hours > 23 or offset_minutes > 59:
                return None
            local_as_utc = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc)
            offset = timedelta(hours=offset_hours, minutes=offset_minutes)
            if sign == "-":
                offset = -offset
            utc_dt = local_as_utc - offset
            return _datetime_to_ms(utc_dt)
        except (ValueError, OverflowError):
            return None

    if RE_DIGITS.fullmatch(value):
        # F3 is an unsigned digit string representing signed-int64 epoch ms.
        try:
            result = int(value)
        except ValueError:
            return None
        if result > INT64_MAX:
            return None
        return result

    return None


def strip_lf(line):
    """Remove one LF only; deliberately preserve a preceding CR."""
    return line[:-1] if line.endswith("\n") else line


def parse_csv_line(line, n_fields):
    """Parse the challenge's restricted CSV dialect, preserving empty fields.

    Quotes may surround a whole field only. There are no escaped quotes,
