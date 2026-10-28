"""土壌（肥沃度・水はけ）と用水（SPEC 第4.4章）。フェーズ2で追加。"""

from __future__ import annotations

import numpy as np

from .. import kernels
from .. import rng as rngmod
from ..grid import Grid
from ..noise import fbm


def _slope(h: np.ndarray, dx: float) -> np.ndarray:
    gy, gx = np.gradient(h, dx)
    return np.hypot(gx, gy)


def make_soil(seed: int, scfg, gcfg, grid: Grid, h, sea, bedrock, alluvium_m) -> tuple[np.ndarray, np.ndarray]:
    """肥沃度と水はけ（0〜1）。"""
    names = [rock.name for rock in gcfg.rocks]
    alluvial = alluvium_m > float(gcfg.alluvium.min_thickness_m)
    s = _slope(h, grid.dx)

    f = scfg.fertility
    base = np.array([float(f.rock[name]) for name in names])[bedrock]
    base = np.where(alluvial, float(f.alluvium), base)
    noise = fbm(grid, rngmod.stream(seed, "soil.fertility"), float(f.noise_scale_km), int(f.octaves), float(f.persistence))
    fert = base * np.exp(-s / float(f.slope_scale)) * (1.0 + float(f.noise_amplitude) * noise)
    fert = np.where(sea, 0.0, np.clip(fert, 0.0, 1.0))

    d = scfg.drainage
    dbase = np.array([float(d.rock[name]) for name in names])[bedrock]
    dbase = np.where(alluvial, float(d.alluvium), dbase)
    drain = np.where(sea, 0.0, np.clip(dbase + float(d.slope_gain) * s, 0.0, 1.0))
    return fert.astype(np.float32), drain.astype(np.float32)


def make_irrigation(icfg, grid: Grid, h, sea, low_flow) -> np.ndarray:
    """用水を引ける度合い（0〜1）。渇水期でも水がある川のマスから、距離と高さの差で決まる。"""
    src = (low_flow >= float(icfg.min_low_flow_m3s)) & ~sea
    dist, nearest = kernels.nearest_source(src.ravel(), grid.nx, grid.ny, grid.dx)
    dist = dist.reshape(h.shape)
    nearest = nearest.reshape(h.shape)
    ok = nearest >= 0
    head = np.where(ok, h - h.ravel()[np.maximum(nearest, 0)], np.inf)
    reach = float(icfg.max_distance_km) * 1000.0
    w = np.exp(-dist / reach) * (head <= float(icfg.max_head_m))
    return np.where(ok & ~sea, w, 0.0).astype(np.float32)
