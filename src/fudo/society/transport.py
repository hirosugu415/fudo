"""移動と運搬のコスト、最小コスト経路、通行量、道の自然発生（SPEC 6.1、9.7）。

- 人の移動：1マスを横切る時間 [時間]（歩き。道があると速い。大きな川は渡るのに時間がかかる）
- 荷物の運搬：1kg を1マス運ぶ費用 [円/kg]（人が担ぐ・荷車・舟のうち最も安いもの）
道は、人や荷物がよく通るマスが踏み固められてできる。上から道を引く処理はない。
"""

from __future__ import annotations

import numpy as np

from .. import kernels
from ..calendar import MEAN_DAYS


def annual_mean_snow_mm(T_norm, P_norm, ccfg, hcfg) -> np.ndarray:
    """平年値の気候で2年間、積雪を度日法で計算し、2年目の月末の積雪の平均 [mm] を返す。"""
    thr = float(ccfg.precipitation.snow_threshold_c)
    tm = float(hcfg.snow.melt_threshold_c)
    ddf = float(hcfg.snow.degree_day_mm_per_c_day)
    swe = np.zeros(T_norm.shape[1:])
    acc = np.zeros_like(swe)
    for y in range(2):
        for mi in range(12):
            T = T_norm[mi]
            swe = swe + np.where(T < thr, P_norm[mi], 0.0)
            swe = np.maximum(swe - np.where(T > tm, ddf * (T - tm) * MEAN_DAYS[mi], 0.0), 0.0)
            if y == 1:
                acc += swe
    return acc / 12.0


class Transport:
    def __init__(self, arrays: dict, dx: float, tcfg, outside, ccfg, hcfg):
        self.cfg = tcfg
        self.dx = dx
        h = arrays["elevation"]
        self.ny, self.nx = h.shape
        gy, gx = np.gradient(h, dx)
        self.slope = np.hypot(gx, gy).ravel()
        self.passable = (~arrays["sea"] & ~arrays["lake"]).ravel()
        q = arrays["mean_discharge_m3s"].ravel()
        self.q = q
        b = tcfg.boat
        self.navigable = self.passable & (q >= float(b.min_discharge_m3s)) & (self.slope <= float(b.max_slope))
        cr = tcfg.crossing
        qf = float(cr.ford_max_discharge_m3s)
        self.crossing_hours = np.where(
            self.passable & (q > qf), float(cr.wait_hours_per_log_q) * np.log(np.maximum(q, qf) / qf), 0.0
        )
        self.snow = annual_mean_snow_mm(arrays["T_norm"], arrays["P_norm"], ccfg, hcfg).ravel()
        self.vot = float(outside.wage_yen_per_day) / float(outside.work_hours_per_day)   # 円/時間
        self.person_hours = np.zeros(self.nx * self.ny)
        self.freight_per_kg = np.zeros(self.nx * self.ny)

    def update_costs(self, road: np.ndarray) -> None:
        w = self.cfg.walk
        s = self.slope
        a = float(w.slope_coef)
        off = float(w.slope_offset)
        v = float(w.speed_max_kmh) * 0.5 * (np.exp(-a * np.abs(s + off)) + np.exp(-a * np.abs(-s + off)))
        off = float(w.offroad_speed_factor)
        v = v * (off + (1.0 - off) * road) / (1.0 + self.snow / float(w.snow_ref_mm))
        km = self.dx / 1000.0
        t_walk = km / v + self.crossing_hours
        self.person_hours = np.where(self.passable, t_walk, np.inf)

        porter = t_walk * self.vot / float(self.cfg.porter.load_kg)
        c = self.cfg.cart
        cart_ok = (road >= float(c.min_road_level)) & (s <= float(c.max_slope))
        cart = np.where(cart_ok, t_walk / float(c.speed_factor) * (self.vot + float(c.cost_yen_per_hour)) / float(c.load_kg), np.inf)
        b = self.cfg.boat
        boat = np.where(self.navigable, km / float(b.speed_kmh) * float(b.crew) * self.vot / float(b.load_kg), np.inf)
        f = np.minimum(np.minimum(porter, cart), boat)
        self.freight_per_kg = np.where(self.passable, f, np.inf)

    def person_tree(self, source: int, cutoff: float):
        return kernels.dijkstra(self.person_hours, self.nx, self.ny, int(source), float(cutoff))

    def freight_tree(self, source: int):
        return kernels.dijkstra(self.freight_per_kg, self.nx, self.ny, int(source), np.inf)


def update_roads(road: np.ndarray, traffic: np.ndarray, rcfg, years: float) -> np.ndarray:
    """道の状態を、通行量から決まる目標に近づける（踏み固められる・荒れる）。"""
    half = float(rcfg.half_traffic_trips_per_year)
    target = traffic / (traffic + half)
    rate = 1.0 - np.exp(-float(rcfg.adapt_rate_per_year) * years)
    return road + rate * (target - road)


def path_cells(pred: np.ndarray, end: int) -> np.ndarray:
    """pred をたどって end から源までのセル列（end が先頭）。"""
    out = []
    i = int(end)
    guard = 0
    while i >= 0 and guard < pred.size:
        out.append(i)
        i = int(pred[i])
        guard += 1
    return np.array(out, dtype=np.int64)


def traveler_stops(path: np.ndarray, person_hours: np.ndarray, elevation: np.ndarray,
                   crossing_hours: np.ndarray, icfg, dx: float) -> list[int]:
    """旅人が経路の上で休む・泊まるマス。

    旅人の行動のルール：
    - rest_interval_hours 歩くごとに休む（茶屋）。lodging_interval_hours ごとに泊まる（宿場）
    - この先 climb_lookahead_km の間に climb_rest_m 以上登るなら、登る前に休む（峠のふもと）
    - 渡るのに crossing_rest_hours 以上かかる川の手前で休む（渡し場）
    """
    stops = []
    rest_int = float(icfg.rest_interval_hours)
    lodge_int = float(icfg.lodging_interval_hours)
    look = max(1, int(round(float(icfg.climb_lookahead_km) * 1000.0 / dx)))
    climb = float(icfg.climb_rest_m)
    cross = float(icfg.crossing_rest_hours)
    since_rest = 0.0
    since_lodge = 0.0
    climbing = False
    for k in range(path.size):
        c = path[k]
        t = person_hours[c]
        rest_here = False
        nxt = path[k + 1] if k + 1 < path.size else -1
        if nxt >= 0 and crossing_hours[nxt] >= cross:
            rest_here = True
        ahead = path[k: k + look + 1]
        rise = elevation[ahead].max() - elevation[c] if ahead.size else 0.0
        if rise >= climb and not climbing:
            rest_here = True
            climbing = True
        elif rise < climb:
            climbing = False
        since_rest += t
        since_lodge += t
        if since_lodge >= lodge_int:
            rest_here = True
            since_lodge = 0.0
        if since_rest >= rest_int:
            rest_here = True
        if rest_here:
            stops.append(int(c))
            since_rest = 0.0
    return stops
