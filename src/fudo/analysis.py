"""分析：SPEC 第14章の現象が起きているかを測る（テストと、後のフェーズの年代記で使う）。結果には影響しない。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import yaml

from . import kernels
from .config import Section, project_root


def load_analysis_config(path: Path | None = None) -> Section:
    p = Path(path) if path else project_root() / "config" / "analysis.yaml"
    with open(p, encoding="utf-8") as f:
        return Section(yaml.safe_load(f), "analysis")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def settlements(households: np.ndarray, population: np.ndarray, dense: int) -> list[dict]:
    """集落：世帯が dense 以上のマスが8近傍でつながったまとまり。人口の多い順。"""
    ny, nx = households.shape
    mask = households >= dense
    seen = np.zeros_like(mask)
    out = []
    for start in np.flatnonzero(mask.ravel()):
        if seen.flat[start]:
            continue
        q = deque([int(start)])
        seen.flat[start] = True
        cells = []
        while q:
            i = q.popleft()
            cells.append(i)
            r, c = divmod(i, nx)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ny and 0 <= cc < nx and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append(rr * nx + cc)
        cells = np.array(cells)
        pop = float(population.ravel()[cells].sum())
        center = int(cells[np.argmax(households.ravel()[cells])])
        out.append({"cells": cells, "population": pop, "center": center})
    out.sort(key=lambda s: -s["population"])
    return out


def rank_size_slope(sett: list[dict], top_n: int) -> float:
    sizes = np.array([s["population"] for s in sett[:top_n]])
    ranks = np.arange(1, sizes.size + 1)
    return float(np.polyfit(np.log(ranks), np.log(sizes), 1)[0])


def market_town_features(arrays: dict, road: np.ndarray, acfg, road_visible: float, ford_max_q: float, dx: float) -> dict:
    """合流点・渡し場・峠のふもとのマス（種類ごとの bool の格子）。"""
    m = acfg.market_towns
    q = arrays["mean_discharge_m3s"]
    ny, nx = q.shape
    land = ~arrays["sea"] & ~arrays["lake"]
    rcv = arrays["receiver"].astype(np.int64)
    qf = q.ravel()
    trib = (qf >= float(m.tributary_discharge_m3s)) & land.ravel()
    donors = np.bincount(rcv[trib & (rcv >= 0)], minlength=qf.size)
    confluence = (donors >= 2) & (qf >= float(m.river_discharge_m3s))
    ford = (road.ravel() >= road_visible) & (qf > ford_max_q) & land.ravel()
    h = arrays["elevation"]
    r = max(1, int(round(float(m.pass_radius_km) * 1000.0 / dx)))
    pad = np.pad(h, r, mode="edge")
    hmax = np.full(h.shape, -np.inf)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr * dr + dc * dc <= r * r:
                hmax = np.maximum(hmax, pad[r + dr: r + dr + ny, r + dc: r + dc + nx])
    gy, gx = np.gradient(h, dx)
    flat = np.hypot(gx, gy) < float(m.foot_max_slope)
    pass_foot = ((hmax - h) >= float(m.pass_climb_m)) & (road >= road_visible) & land & flat
    return {"confluence": confluence.reshape(ny, nx), "ford": ford.reshape(ny, nx), "pass_foot": pass_foot}


def near_mask(mask: np.ndarray, km: float, dx: float) -> np.ndarray:
    ny, nx = mask.shape
    d = kernels.distance_from(mask.ravel(), nx, ny, dx).reshape(ny, nx)
    return d <= km * 1000.0


def households_near(households: np.ndarray, cell: int, radius_km: float, dx_m: float) -> float:
    """マス cell から半径 radius_km 以内の世帯数（鉱山町の大きさを測る）。"""
    ny, nx = households.shape
    r0, c0 = divmod(int(cell), nx)
    rr, cc = np.mgrid[0:ny, 0:nx]
    near = np.hypot(rr - r0, cc - c0) * dx_m / 1000.0 <= radius_km
    return float(households[near].sum())
