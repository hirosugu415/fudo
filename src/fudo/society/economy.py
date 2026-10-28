"""外部世界モデル（SPEC 第8章）。フィールドの外を、年ごとの集計値で表す。

外部世界はフィールドのことを知らず、フィールドの出来事にも影響されない（フィールドは外の経済に比べて十分小さい）。
そのため系列は、実行の始めに起点の年（1700年）から毎年まとめて計算できる。開始年やウォームアップの長さを変えても、
同じ年の値は同じ（気候のゆらぎと同じ考え方）。景気の乱数は世界シードの名前付きストリームから取る。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import rng as rngmod

GOODS = ("rice", "fish", "coal", "iron_ore", "copper", "limestone")


@dataclass
class Year:
    productivity: float           # 1人あたりの産出量（基準の年 = 1）
    cycle: float                  # 景気（トレンドからの乖離の割合）
    wage_yen_per_day: float
    interest: float               # 1/年
    city_pop: np.ndarray          # 大都市ごとの人口 [人]
    prices: dict                  # 品物ごとの外部の価格 [円/t]


class External:
    def __init__(self, world_seed: int, info_external: dict, ocfg, end_year: int):
        o = ocfg
        self.cfg = o
        self.origin = int(o.origin_year)
        anchor = int(o.anchor_year)
        amp = float(info_external["business_cycle_amplitude"])
        self.amplitude = amp
        g_pop = float(info_external["population_growth_per_year"])
        pop_anchor = np.array([c["population"] for c in info_external["cities"]], dtype=np.float64)
        years = np.arange(self.origin, max(end_year, anchor) + 1)
        n = years.size

        # 景気：AR(1)。毎年1個ずつ同じ順で引く
        rho = float(o.business_cycle.persistence)
        eps = rngmod.stream(world_seed, "external.cycle").standard_normal(n) * amp * np.sqrt(1.0 - rho * rho)
        x = np.zeros(n)
        x[0] = eps[0] / np.sqrt(1.0 - rho * rho)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + eps[i]

        # 生産性：基準の年で1
        pc = o.productivity
        g = np.where(years < anchor, float(pc.growth_before_anchor_per_year), float(pc.growth_after_anchor_per_year))
        logA = np.concatenate([[0.0], np.cumsum(np.log1p(g[:-1]))])
        logA -= logA[years == anchor][0]
        A = np.exp(logA)

        # 人口：伸びは所得が上がると鈍る（人口転換）。基準の年に生成時の人口になるようにする
        rate = g_pop * np.exp(-np.maximum(A - 1.0, 0.0) / float(o.population.transition_productivity))
        logP = np.concatenate([[0.0], np.cumsum(np.log1p(rate[:-1]))])
        logP -= logP[years == anchor][0]
        P = np.exp(logP)            # 基準の年に対する倍率（どの都市も同じ伸び）

        # 価格：需要と、遅れて反応する供給の能力
        income = A * (1.0 + x)
        prices = {}
        for name in GOODS:
            gc = o.goods[name]
            base = float(gc.base_yen_per_t)
            D = P * (A * (1.0 + float(gc.cycle_elasticity) * x)).clip(1e-6) ** float(gc.income_elasticity)
            inv_e = 1.0 / float(gc.demand_elasticity)
            k = float(gc.supply_response)
            K = D[0]
            p = np.zeros(n)
            for i in range(n):
                p[i] = base * (D[i] / K) ** inv_e
                K = K * np.exp(k * np.log(p[i] / base))
            prices[name] = p

        ic = o.interest
        r = float(ic.base_per_year) * (1.0 + float(ic.cycle_sensitivity) * x / max(amp, 1e-12))
        self.years = years
        self.A = A
        self.x = x
        self.wage = float(o.wage_yen_per_day) * income
        self.interest = np.maximum(r, float(ic.min_per_year))
        self.pop = P[:, None] * pop_anchor[None, :]
        self.prices = prices

    def at(self, year: int) -> Year:
        i = int(year) - self.origin
        if not 0 <= i < self.years.size:
            raise ValueError(f"外部世界の系列の範囲外の年です: {year}")
        return Year(float(self.A[i]), float(self.x[i]), float(self.wage[i]), float(self.interest[i]),
                    self.pop[i].copy(), {k: float(v[i]) for k, v in self.prices.items()})
