"""資源：鉱物・森林・漁場（SPEC 第4.4章）。"""

from __future__ import annotations

import numpy as np

from .. import kernels
from .. import rng as rngmod
from ..grid import Grid


def make_minerals(seed: int, mcfg, gcfg, grid: Grid, bedrock, alluvium_m, land) -> tuple[dict, list[dict]]:
    """鉱物の種類ごとの埋蔵量の格子 [t/マス] と、鉱床の一覧を返す。"""
    rock_index = {rock.name: i for i, rock in enumerate(gcfg.rocks)}
    ny, nx = bedrock.shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    rasters: dict[str, np.ndarray] = {}
    deposits: list[dict] = []
    for dep in mcfg.deposits:
        r = rngmod.stream(seed, f"minerals.{dep.name}")
        host = np.isin(bedrock, [rock_index[n] for n in dep.host_rocks]) & land
        if dep.contact_with:
            contact = np.isin(bedrock, [rock_index[n] for n in dep.contact_with])
            d = kernels.distance_from(contact.ravel(), nx, ny, grid.dx).reshape(ny, nx)
            host &= d <= float(dep.contact_distance_km) * 1000.0
        cand = np.flatnonzero(host.ravel())
        raster = np.zeros((ny, nx), dtype=np.float64)
        area_km2 = cand.size * grid.cell_area / 1e6
        count = int(r.poisson(float(dep.density_per_100km2) * area_km2 / 100.0)) if cand.size else 0
        for _ in range(count):
            center = int(cand[int(r.integers(0, cand.size))])
            radius = float(dep.radius_km.median) * float(np.exp(float(dep.radius_km.log_sd) * r.standard_normal()))
            reserve = float(dep.reserve_t.median) * float(np.exp(float(dep.reserve_t.log_sd) * r.standard_normal()))
            g = dep.grade
            grade = float(np.clip(float(g.mean) + float(g.sd) * r.standard_normal(), float(g.min), float(g.max)))
            top = float(r.uniform(float(dep.depth_m.min), float(dep.depth_m.max)))
            cr, cc = divmod(center, nx)
            dist = np.hypot(rows - cr, cols - cc) * grid.dx
            foot = (dist <= max(radius * 1000.0, grid.dx * 0.5)) & host
            cells = int(foot.sum())
            raster[foot] += reserve / cells
            deposits.append({
                "mineral": dep.name,
                "label": dep.label,
                "row": int(cr),
                "col": int(cc),
                "cells": cells,
                "reserve_t": reserve,
                "grade": grade,
                "depth_top_m": top + float(alluvium_m[cr, cc]),
            })
        rasters[dep.name] = raster
    return rasters, deposits


def make_forest(fcfg, normals, land_nonlake) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """森林の型（-1 は森林なし）、蓄積量の上限 [m³/マス]、内的増加率 [1/年]。初期の蓄積は上限とする。"""
    base = float(fcfg.base_c)
    T = normals.T
    wi = np.maximum(T - base, 0.0).sum(axis=0)
    ci = np.minimum(T - base, 0.0).sum(axis=0)
    p_ann = normals.P.sum(axis=0)
    ftype = np.full(wi.shape, -1, dtype=np.int8)
    for k, t in enumerate(fcfg.types):
        ok = (ftype < 0) & (wi >= float(t.wi_min)) & (wi < float(t.wi_max)) & (ci >= float(t.ci_min))
        ftype[ok] = k
    ftype[~land_nonlake | (p_ann < float(fcfg.min_annual_precip_mm))] = -1
    stock_tab = np.array([float(t.max_stock_m3_per_ha) for t in fcfg.types])
    rate_tab = np.array([float(t.growth_rate_per_yr) for t in fcfg.types])
    return ftype, stock_tab, rate_tab


def make_fishery(fcfg, grid: Grid, h, sea, lakes, mean_q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """漁場の種類（-1 なし、0 沿岸、1 湖、2 川）、資源量の上限 [t/マス]、内的増加率 [1/年]。"""
    kind = np.full(h.shape, -1, dtype=np.int8)
    coastal = sea & (h > -float(fcfg.coastal.max_depth_m))
    kind[coastal] = 0
    kind[lakes] = 1
    river = (~sea) & (~lakes) & (mean_q > float(fcfg.river.min_discharge_m3s))
    kind[river] = 2
    km2 = grid.cell_area / 1e6
    cap_tab = np.array([float(fcfg.coastal.capacity_t_per_km2), float(fcfg.lake.capacity_t_per_km2),
                        float(fcfg.river.capacity_t_per_km2)]) * km2
    rate_tab = np.array([float(fcfg.coastal.growth_rate_per_yr), float(fcfg.lake.growth_rate_per_yr),
                         float(fcfg.river.growth_rate_per_yr)])
    cap = np.where(kind >= 0, cap_tab[np.maximum(kind, 0)], 0.0)
    rate = np.where(kind >= 0, rate_tab[np.maximum(kind, 0)], 0.0)
    return kind, cap, rate
