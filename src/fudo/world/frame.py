"""世界の枠組み：緯度、海の端、外部世界の地形構成、端の外側の地形、風。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import rng as rngmod
from ..grid import EDGES, Grid


@dataclass
class Frame:
    latitude: float
    composition: dict[str, float]           # 外部世界の地形構成 {mountain, plain, island}
    edge_kind: dict[str, str]               # 端ごとの外側：sea / mountain / plain
    outer_elevation: dict[str, float]       # 端ごとの外側の標高 [m]（海は境界の水深）
    outer_sea_km: dict[str, float]          # 端の外側から海までの距離 [km]（海の端は0）
    moisture_factor: dict[str, float]       # 端から入る空気の湿り気の倍率（距離による減衰を除く）
    gorge: dict | None                      # どの端も開いていないときの出口 {edge, start, width}
    westerly_speed: float                   # m/s
    monsoon_speed: float                    # m/s
    monsoon_summer_from_deg: float          # 度
    rain_systems: dict[str, float]          # 向きによらない雨の最盛期の上乗せ量 [mm] {baiu, typhoon}
    wind: list[tuple[float, float, float]] = field(default_factory=list)  # 月ごとの (wx, wy, 風速)


def _choose_sea_edges(cfg, r: np.random.Generator) -> list[str]:
    probs = np.asarray(cfg.count_probabilities, dtype=np.float64)
    count = int(r.choice(len(probs), p=probs / probs.sum()))
    if count == 0:
        return []
    first = int(r.integers(0, 4))
    if count == 1:
        return [EDGES[first]]
    if r.random() < float(cfg.adjacent_probability):
        second = (first + 1) % 4
    else:
        second = (first + 2) % 4
    return sorted({EDGES[first], EDGES[second]}, key=EDGES.index)


def monthly_wind(frame: Frame, fcfg) -> list[tuple[float, float, float]]:
    """月ごとの風（向きの単位ベクトルと風速）。列方向が東、行方向が南。"""
    w = fcfg.wind
    ramp = w.westerly_lat_ramp_deg
    wlat = np.clip((frame.latitude - float(ramp["from"])) / (float(ramp.to) - float(ramp["from"])), 0.0, 1.0)
    b = np.deg2rad(frame.monsoon_summer_from_deg)
    out = []
    for month in range(1, 13):
        s = np.cos(2.0 * np.pi * (month - float(w.monsoon_peak_month)) / 12.0)
        # 東向き成分・北向き成分（「方位 b から吹く」風は b + 180° の向きに進む）
        east = frame.westerly_speed * wlat + frame.monsoon_speed * s * (-np.sin(b))
        north = frame.monsoon_speed * s * (-np.cos(b))
        speed = float(np.hypot(east, north))
        if speed < 1e-9:
            east, north, speed = 1.0, 0.0, 1.0
        wx = float(east / speed)
        wy = float(-north / speed)
        out.append((wx, wy, max(speed, float(w.min_speed_ms))))
    return out


def make_frame(seed: int, gen, grid: Grid) -> Frame:
    fcfg = gen.frame
    ecfg = gen.external
    ccfg = gen.climate

    r = rngmod.stream(seed, "frame.latitude")
    lat = float(r.uniform(float(fcfg.latitude_deg.min), float(fcfg.latitude_deg.max)))

    sea_edges = _choose_sea_edges(fcfg.sea_edges, rngmod.stream(seed, "frame.sea_edges"))

    r = rngmod.stream(seed, "external.composition")
    comp = r.dirichlet(np.asarray(ecfg.composition.dirichlet_alpha, dtype=np.float64))
    composition = {"mountain": float(comp[0]), "plain": float(comp[1]), "island": float(comp[2])}

    r = rngmod.stream(seed, "frame.edges")
    p_mountain = composition["mountain"] / (composition["mountain"] + composition["plain"])
    mcfg = ccfg.moisture
    ed = ecfg.edges
    edge_kind, outer_elev, outer_km, moist = {}, {}, {}, {}
    for e in EDGES:
        # 端ごとに同じ回数だけ乱数を引く（端の種類で乱数列がずれないように）
        u_kind, u_elev, z_dist, u_mtn, z_noise = (
            r.random(), r.random(), r.standard_normal(), r.random(), r.standard_normal()
        )
        noise = float(np.exp(float(mcfg.edge_log_sd) * z_noise))
        if e in sea_edges:
            edge_kind[e] = "sea"
            outer_elev[e] = float(gen.erosion.numerics.sea_edge_depth_m)
            outer_km[e] = 0.0
            moist[e] = noise
            continue
        kind = "mountain" if u_kind < p_mountain else "plain"
        edge_kind[e] = kind
        rng_e = ed.mountain_outer_elevation_m if kind == "mountain" else ed.plain_outer_elevation_m
        outer_elev[e] = float(rng_e.min) + u_elev * (float(rng_e.max) - float(rng_e.min))
        d = float(ed.outer_sea_distance_km.median) * float(np.exp(float(ed.outer_sea_distance_km.log_sd) * z_dist))
        d *= 1.0 - float(ed.island_distance_shortening) * composition["island"]
        outer_km[e] = d
        f = 1.0
        if kind == "mountain":
            mf = mcfg.mountain_edge_factor
            f = float(mf.min) + u_mtn * (float(mf.max) - float(mf.min))
        moist[e] = f * noise

    gorge = None
    if all(edge_kind[e] == "mountain" for e in EDGES):
        # 海も平地の端もないときは、山地の端の1か所に出口（峡谷）を作る
        r = rngmod.stream(seed, "frame.gorge")
        e = EDGES[int(r.integers(0, 4))]
        length = grid.nx if e in ("north", "south") else grid.ny
        width = int(ed.outlet_gorge_width_cells)
        start = int(r.integers(length // 4, 3 * length // 4 - width))
        pe = ed.plain_outer_elevation_m
        elev = float(pe.min) + float(r.random()) * (float(pe.max) - float(pe.min))
        gorge = {"edge": e, "start": start, "width": width, "elevation": elev}

    r = rngmod.stream(seed, "climate.rain_systems")
    rs = ccfg.rain_systems
    rain_systems = {
        name: float(r.uniform(float(rs[name].amount_mm.min), float(rs[name].amount_mm.max)))
        for name in ("baiu", "typhoon")
    }

    r = rngmod.stream(seed, "frame.wind")
    w = fcfg.wind
    frame = Frame(
        latitude=lat,
        composition=composition,
        edge_kind=edge_kind,
        outer_elevation=outer_elev,
        outer_sea_km=outer_km,
        moisture_factor=moist,
        gorge=gorge,
        rain_systems=rain_systems,
        westerly_speed=float(r.uniform(float(w.westerly_speed_ms.min), float(w.westerly_speed_ms.max))),
        monsoon_speed=float(r.uniform(float(w.monsoon_speed_ms.min), float(w.monsoon_speed_ms.max))),
        monsoon_summer_from_deg=float(
            r.uniform(float(w.monsoon_summer_from_deg.min), float(w.monsoon_summer_from_deg.max))
        ),
    )
    frame.wind = monthly_wind(frame, fcfg)
    return frame
