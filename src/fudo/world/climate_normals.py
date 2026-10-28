"""気候の平年値（12か月分）：気温、降水（水蒸気の移流と地形性降雨）、可能蒸発散。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import kernels
from ..calendar import MEAN_DAYS, SECONDS_PER_DAY, mid_month_day_of_year
from ..grid import EDGES, Grid
from .frame import Frame


@dataclass
class ClimateNormals:
    T: np.ndarray        # (12, ny, nx) ℃
    P: np.ndarray        # (12, ny, nx) mm/月（平均日数の月）
    PET: np.ndarray      # (12, ny, nx) mm/月（平均日数の月）
    sea_dist_km: np.ndarray  # (ny, nx)
    q_edge: np.ndarray   # (12, 4) 端から入る空気の水蒸気量 [mm]
    T_sea_level: np.ndarray  # (12,) 海面での気温（海岸の値）℃
    daylen_h: np.ndarray     # (12,) 日の長さ [時間]


def saturation_vapor_hpa(T, cfg):
    s = cfg.saturation_vapor
    return float(s.a_hpa) * np.exp(float(s.b) * T / (T + float(s.c_c)))


def day_length_hours(lat: float, cfg) -> np.ndarray:
    e = cfg.evapotranspiration
    phi = np.deg2rad(lat)
    out = np.empty(12)
    for m in range(1, 13):
        doy = mid_month_day_of_year(m)
        decl = np.deg2rad(float(e.obliquity_deg)) * np.sin(2.0 * np.pi * (float(e.declination_day_offset) + doy) / 365.0)
        x = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
        out[m - 1] = 24.0 * np.arccos(x) / np.pi
    return out


def pet_mm(T, daylen_h, days, cfg):
    e = cfg.evapotranspiration
    s = cfg.saturation_vapor
    return kernels.hamon_pet_mm(
        T, daylen_h, days, float(e.coefficient), float(e.hamon_constant),
        float(s.a_hpa), float(s.b), float(s.c_c), float(e.vapor_density_factor), float(e.vapor_density_offset_c),
    )


def sea_level_temperature(lat: float, month: float, sea_dist_km, cfg):
    t = cfg.temperature
    mean = float(t.annual_mean.at_lat_35_c) + float(t.annual_mean.per_deg_lat_c) * (lat - 35.0)
    half = float(t.half_annual_range.at_lat_35_c) + float(t.half_annual_range.per_deg_lat_c) * (lat - 35.0)
    c = t.continentality
    cont = 1.0 - np.exp(-np.asarray(sea_dist_km) / float(c.length_km))
    half = half + float(c.extra_half_range_c) * cont
    mean = mean + float(c.mean_shift_c) * cont
    return mean + half * np.cos(2.0 * np.pi * (month - float(t.warmest_month)) / 12.0)


def sea_surface_temperature(lat: float, month: float, cfg):
    t = cfg.temperature
    s = cfg.sea_surface
    mean = float(t.annual_mean.at_lat_35_c) + float(t.annual_mean.per_deg_lat_c) * (lat - 35.0)
    half = float(t.half_annual_range.at_lat_35_c) + float(t.half_annual_range.per_deg_lat_c) * (lat - 35.0)
    return mean + float(s.mean_offset_c) + half * float(s.amplitude_ratio) * np.cos(
        2.0 * np.pi * (month - float(t.warmest_month) - float(s.lag_months)) / 12.0
    )


def nondirectional_rain_mm(frame: Frame, month: int, T_month, cfg):
    """季節風の向きによらない雨（梅雨前線・台風・夕立）の月降水量 [mm/平均日数の月]。標高による増加は含めない。"""
    rs = cfg.rain_systems
    total = 0.0
    for name in ("baiu", "typhoon"):
        s = rs[name]
        lat_w = np.clip(
            (float(s.lat_zero_deg) - frame.latitude) / (float(s.lat_zero_deg) - float(s.lat_full_deg)), 0.0, 1.0
        )
        shape = np.exp(-0.5 * ((month - float(s.peak_month)) / float(s.width_months)) ** 2)
        total = total + frame.rain_systems[name] * lat_w * shape
    cv = rs.convective
    return total + float(cv.mm_per_c) * np.maximum(np.asarray(T_month) - float(cv.base_c), 0.0)


def sea_distance_km(grid: Grid, sea: np.ndarray, frame: Frame) -> np.ndarray:
    """各マスから最も近い海までの距離 [km]。マップ外の海（端の外側の距離）も考える。"""
    d = kernels.distance_from(sea.ravel(), grid.nx, grid.ny, grid.dx).reshape(grid.ny, grid.nx)
    for e in EDGES:
        if frame.edge_kind[e] == "sea":
            continue
        d = np.minimum(d, grid.distance_to_edge_m(e) + frame.outer_sea_km[e] * 1000.0)
    return d / 1000.0


def _smooth(a: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return a
    k = 2 * radius + 1
    pad = np.pad(a, radius, mode="edge")
    cs = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    s = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
    return s / (k * k)


def compute_normals(grid: Grid, h: np.ndarray, sea: np.ndarray, frame: Frame, cfg) -> ClimateNormals:
    lat = frame.latitude
    land_h = np.where(sea, 0.0, np.maximum(h, 0.0))
    dist_km = sea_distance_km(grid, sea, frame)
    daylen = day_length_hours(lat, cfg)
    lapse = float(cfg.temperature.lapse_rate_c_per_km) / 1000.0

    m_cfg = cfg.moisture
    p_cfg = cfg.precipitation
    pw_ref = float(m_cfg.precipitable_water_ref_mm)
    es_ref = saturation_vapor_hpa(float(m_cfg.reference_temp_c), cfg)

    hs = _smooth(land_h, int(p_cfg.slope_smoothing_cells))
    dh_drow, dh_dcol = np.gradient(hs, grid.dx)

    T = np.empty((12, grid.ny, grid.nx))
    P = np.empty((12, grid.ny, grid.nx))
    PET = np.empty((12, grid.ny, grid.nx))
    q_edge = np.empty((12, 4))
    T_sl = np.empty(12)
    sea_flat = sea.ravel()
    for mi in range(12):
        month = mi + 1
        T[mi] = sea_level_temperature(lat, month, dist_km, cfg) - lapse * land_h
        T_sl[mi] = float(sea_level_temperature(lat, month, 0.0, cfg))
        sst = float(sea_surface_temperature(lat, month, cfg))
        q_sea = pw_ref * float(saturation_vapor_hpa(sst, cfg)) / es_ref
        for k, e in enumerate(EDGES):
            if frame.edge_kind[e] == "sea":
                q_edge[mi, k] = q_sea * frame.moisture_factor[e]
            else:
                d = frame.outer_sea_km[e]
                t_out = float(sea_level_temperature(lat, month, d, cfg))
                q_edge[mi, k] = (
                    pw_ref * float(saturation_vapor_hpa(t_out, cfg)) / es_ref
                    * np.exp(-d / float(m_cfg.land_decay_km)) * frame.moisture_factor[e]
                )
        wx, wy, speed = frame.wind[mi]
        slope_w = (dh_dcol * wx + dh_drow * wy).ravel()
        seconds = MEAN_DAYS[mi] * SECONDS_PER_DAY
        P[mi] = kernels.upwind_precipitation(
            grid.nx, grid.ny, grid.dx, wx, wy, speed, slope_w, sea_flat, q_edge[mi], q_sea,
            float(p_cfg.rainout_length_km) * 1000.0, float(p_cfg.orographic_gain),
            float(p_cfg.orographic_max_factor), float(p_cfg.lee_suppression),
            float(p_cfg.lee_floor), float(m_cfg.sea_relaxation_km) * 1000.0, seconds,
        ).reshape(grid.ny, grid.nx)
        # 季節風の向きによらない雨を加える（山地ほど多い）
        P[mi] += nondirectional_rain_mm(frame, month, T[mi], cfg) * (
            1.0 + float(cfg.rain_systems.orographic_per_km) * land_h / 1000.0
        )
        PET[mi] = pet_mm(T[mi], daylen[mi], MEAN_DAYS[mi], cfg)
    return ClimateNormals(T=T, P=P, PET=PET, sea_dist_km=dist_km, q_edge=q_edge, T_sea_level=T_sl, daylen_h=daylen)


def relative_runoff(normals: ClimateNormals, land: np.ndarray, min_rel: float) -> np.ndarray:
    """浸食の重みに使う相対流出量（平年の 降水 − 蒸発散 を陸の平均で割ったもの）。"""
    R = np.maximum(normals.P - normals.PET, 0.0).sum(axis=0)
    mean = float(R[land].mean()) if land.any() else 0.0
    if mean <= 0.0:
        return np.full_like(R, min_rel)
    return np.maximum(R / mean, min_rel)
