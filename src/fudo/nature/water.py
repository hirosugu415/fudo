"""月の水収支：積雪・融雪、蒸発散（Hamon 法）、土壌の水分、地下水（線形貯留）、流量。

マップ内の陸のセルと、外から流れ込む川の外側の集水域（1つの箱として扱う）を同じ式で計算する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import kernels
from ..calendar import MEAN_DAYS, SECONDS_PER_DAY


@dataclass
class WaterStatic:
    """世界から読み込む、変化しない入力。"""
    T_norm: np.ndarray        # (12, n) ℃
    P_norm: np.ndarray        # (12, n) mm/平均日数の月
    land: np.ndarray          # (n,) bool
    order: np.ndarray         # (n,) 流れ先が先に来る順
    rcv: np.ndarray           # (n,) 流れ先（-1 はマップ外）
    cell_area: float          # m²
    daylen_h: np.ndarray      # (12,)
    ext_cell: np.ndarray      # (k,) 外から流れ込む川の入口のセル
    ext_area_m2: np.ndarray   # (k,)
    ext_T_norm: np.ndarray    # (12, k) ℃
    ext_P_norm: np.ndarray    # (12, k) mm/平均日数の月


@dataclass
class WaterState:
    snow: np.ndarray   # (n,) mm（水換算）
    soil: np.ndarray   # (n,) mm
    gw: np.ndarray     # (n,) mm
    ext_snow: np.ndarray
    ext_soil: np.ndarray
    ext_gw: np.ndarray

    @classmethod
    def zeros(cls, n: int, k: int) -> "WaterState":
        z = lambda m: np.zeros(m, dtype=np.float64)  # noqa: E731
        return cls(z(n), z(n), z(n), z(k), z(k), z(k))

    def copy(self) -> "WaterState":
        return WaterState(*(a.copy() for a in (self.snow, self.soil, self.gw, self.ext_snow, self.ext_soil, self.ext_gw)))

    def storage_m3(self, land: np.ndarray, area: float, ext_area: np.ndarray) -> tuple[float, float]:
        inside = float((self.snow[land] + self.soil[land] + self.gw[land]).sum()) * 1e-3 * area
        outside = float(((self.ext_snow + self.ext_soil + self.ext_gw) * ext_area).sum()) * 1e-3
        return inside, outside


def _params(hcfg, ccfg):
    e = ccfg.evapotranspiration
    s = ccfg.saturation_vapor
    return (
        float(ccfg.precipitation.snow_threshold_c), float(hcfg.snow.melt_threshold_c),
        float(hcfg.snow.degree_day_mm_per_c_day), float(hcfg.soil.capacity_mm),
        float(hcfg.groundwater.infiltration_fraction), float(hcfg.groundwater.residence_days),
        float(e.coefficient), float(e.hamon_constant), float(s.a_hpa), float(s.b), float(s.c_c),
        float(e.vapor_density_factor), float(e.vapor_density_offset_c),
    )


class WaterModel:
    def __init__(self, static: WaterStatic, hcfg, ccfg):
        self.s = static
        self.params = _params(hcfg, ccfg)
        k = static.ext_cell.size
        self._ext_order = np.arange(k, dtype=np.int64)
        self._ext_rcv = np.full(k, -1, dtype=np.int64)
        self._ext_land = np.ones(k, dtype=bool)
        self._empty_i = np.zeros(0, dtype=np.int64)
        self._empty_f = np.zeros(0, dtype=np.float64)

    def step(self, state: WaterState, month: int, days: float, dT: float, p_factor: float):
        """1か月進める。state を書き換え、(各セルの流量 [m³/s], 集計値の辞書) を返す。"""
        s = self.s
        mi = month - 1
        scale = days / MEAN_DAYS[mi]
        seconds = days * SECONDS_PER_DAY

        # 外側の集水域（1km² の箱として計算し、面積を掛ける）
        k = s.ext_cell.size
        ext_vol = np.zeros(k, dtype=np.float64)
        ext_precip = 0.0
        if k:
            T_e = s.ext_T_norm[mi] + dT
            P_e = s.ext_P_norm[mi] * scale * p_factor
            flow_e, _ = kernels.water_month(
                T_e, P_e, self._ext_land, self._ext_order, self._ext_rcv, 1.0e6,
                state.ext_snow, state.ext_soil, state.ext_gw, days, s.daylen_h[mi], *self.params,
                self._empty_i, self._empty_f,
            )
            ext_vol = flow_e * (s.ext_area_m2 / 1.0e6)
            ext_precip = float((P_e * s.ext_area_m2).sum()) * 1e-3

        T = s.T_norm[mi] + dT
        P = s.P_norm[mi] * scale * p_factor
        flow, diag = kernels.water_month(
            T, P, s.land, s.order, s.rcv, s.cell_area,
            state.snow, state.soil, state.gw, days, s.daylen_h[mi], *self.params,
            s.ext_cell, ext_vol,
        )
        stats = {
            "precip_m3": float(diag[0]),
            "aet_m3": float(diag[1]),
            "outflow_m3": float(diag[2]),
            "inflow_m3": float(diag[3]),
            "ext_precip_m3": ext_precip,
        }
        return flow / seconds, stats


def spinup(model: WaterModel, years: int) -> tuple[WaterState, np.ndarray, np.ndarray]:
    """平年値の気候で years 年回す。最後の状態と、最後の1年の平均流量・最小の月流量 [m³/s] を返す。"""
    n = model.s.land.size
    state = WaterState.zeros(n, model.s.ext_cell.size)
    mean_q = np.zeros(n)
    min_q = np.zeros(n)
    for y in range(years):
        acc = np.zeros(n)
        low = np.full(n, np.inf)
        for month in range(1, 13):
            q, _ = model.step(state, month, MEAN_DAYS[month - 1], 0.0, 1.0)
            acc += q * MEAN_DAYS[month - 1]
            low = np.minimum(low, q)
        mean_q = acc / sum(MEAN_DAYS)
        min_q = low
    return state, mean_q, min_q
