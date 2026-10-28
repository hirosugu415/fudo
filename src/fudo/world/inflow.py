"""マップの外から流れ込む川。外側が山地の端にだけ置く（hydrology.yaml の inflow の説明を参照）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import rng as rngmod
from ..grid import EDGES, Grid
from .frame import Frame


@dataclass
class Inflow:
    edge: str
    cell: int                # 流入点のセル番号（端のセル）
    area_km2: float          # 外側の集水面積
    mean_elevation_m: float  # 外側の集水域の平均標高


def make_inflows(seed: int, hcfg, grid: Grid, frame: Frame) -> list[Inflow]:
    icfg = hcfg.inflow
    spacing = int(icfg.min_spacing_cells)
    margin = int(icfg.edge_margin_cells)
    out: list[Inflow] = []
    for e in EDGES:
        if frame.edge_kind[e] != "mountain":
            continue
        r = rngmod.stream(seed, f"inflow.{e}")
        count = int(r.poisson(float(icfg.per_mountain_edge_mean)))
        cells = grid.edge_cells(e)
        length = cells.size
        taken: list[int] = []
        if frame.gorge is not None and frame.gorge["edge"] == e:
            taken.append(frame.gorge["start"] + frame.gorge["width"] // 2)
        for _ in range(count):
            pos = int(r.integers(margin, length - margin))
            area = float(icfg.area_km2.median) * float(np.exp(float(icfg.area_km2.log_sd) * r.standard_normal()))
            area = min(area, float(icfg.area_km2.max))
            el = frame.outer_elevation[e] + float(
                r.uniform(float(icfg.mean_elevation_above_edge_m.min), float(icfg.mean_elevation_above_edge_m.max))
            )
            if any(abs(pos - t) < spacing for t in taken):
                continue
            taken.append(pos)
            out.append(Inflow(edge=e, cell=int(cells[pos]), area_km2=area, mean_elevation_m=el))
    return out
