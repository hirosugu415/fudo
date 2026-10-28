"""暦。ステップは月の通し番号 year * 12 + (month - 1) で表す。"""

from __future__ import annotations

# 平年値（12か月分）に使う月の平均日数。グレゴリオ暦の400年平均（2月 = 28 + 97/400）。
MEAN_DAYS = (31.0, 28.2425, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0)
SECONDS_PER_DAY = 86400.0


def parse_year_month(text: str) -> tuple[int, int]:
    y, m = str(text).split("-")
    y, m = int(y), int(m)
    if not 1 <= m <= 12:
        raise ValueError(f"月が不正です: {text}")
    return y, m


def to_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def from_index(index: int) -> tuple[int, int]:
    return index // 12, index % 12 + 1


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if is_leap(year) else 28
    return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]


def mid_month_day_of_year(month: int) -> float:
    """月の中日の年間通日（平年の日数で計算）。"""
    return sum(MEAN_DAYS[: month - 1]) + MEAN_DAYS[month - 1] / 2.0
