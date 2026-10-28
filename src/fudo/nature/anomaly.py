"""年ごとの気候のゆらぎ。固定の起点の年（climate.yaml の anomaly.origin_year）から、年の順に生成する。

ある年の値は、起点からその年までの乱数だけで決まる。シミュレーションの開始年や終了年、
ウォームアップ期間の長さを変えても、同じ世界シードなら同じ年の値は変わらない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import rng as rngmod


@dataclass
class AnomalySeries:
    origin_year: int
    temperature_c: np.ndarray       # (年数,) 気温の平年からのずれ [℃]
    precip_factor: np.ndarray       # (年数,) 降水量の倍率 [無次元]

    def at(self, year: int) -> tuple[float, float]:
        k = year - self.origin_year
        if k < 0 or k >= self.temperature_c.size:
            raise ValueError(f"{year}年の気候のゆらぎは生成範囲（{self.origin_year}年〜）の外です")
        return float(self.temperature_c[k]), float(self.precip_factor[k])


def make_series(world_seed: int, acfg, last_year: int) -> AnomalySeries:
    origin = int(acfg.origin_year)
    if last_year < origin:
        raise ValueError("終了年が気候のゆらぎの起点より前です")
    r = rngmod.stream(world_seed, "climate_anomaly")
    n = last_year - origin + 1
    sd_t, a_t = float(acfg.temperature_sd_c), float(acfg.temperature_ar1)
    sd_p, a_p = float(acfg.precipitation_log_sd), float(acfg.precipitation_ar1)
    t = np.empty(n)
    lp = np.empty(n)
    for k in range(n):
        z = r.standard_normal(2)  # 1年に2個ずつ、毎年同じ順で引く
        if k == 0:
            t[k] = sd_t * z[0]
            lp[k] = sd_p * z[1]
        else:
            t[k] = a_t * t[k - 1] + sd_t * np.sqrt(1.0 - a_t * a_t) * z[0]
            lp[k] = a_p * lp[k - 1] + sd_p * np.sqrt(1.0 - a_p * a_p) * z[1]
    # 対数正規の倍率の平均が1になるように補正する
    factor = np.exp(lp - 0.5 * sd_p * sd_p)
    return AnomalySeries(origin_year=origin, temperature_c=t, precip_factor=factor)
