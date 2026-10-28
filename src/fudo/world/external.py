"""外部世界の性質：大都市の接続点と規模、成長の傾向、景気変動の大きさ（SPEC 第4.6章）。"""

from __future__ import annotations

import numpy as np

from .. import rng as rngmod
from ..grid import EDGES, Grid
from .frame import Frame
from .inflow import Inflow


def _candidates(grid: Grid, frame: Frame, inflows: list[Inflow], h: np.ndarray) -> list[dict]:
    out = []
    for k, inf in enumerate(inflows):
        r, c = divmod(inf.cell, grid.nx)
        out.append({"kind": "river", "edge": inf.edge, "row": r, "col": c, "inflow": k})
    for e in EDGES:
        kind = frame.edge_kind[e]
        cells = grid.edge_cells(e)
        if kind == "sea":
            mid = int(cells[cells.size // 2])
            r, c = divmod(mid, grid.nx)
            out.append({"kind": "sea", "edge": e, "row": r, "col": c})
        elif kind == "plain":
            mid = int(cells[cells.size // 2])
            r, c = divmod(mid, grid.nx)
            out.append({"kind": "plain", "edge": e, "row": r, "col": c})
        else:
            # 峠：山地の端のすぐ内側で最も低い地点（地形と整合させるため、生成後の標高で選ぶ）
            inner = {"north": (1, slice(1, -1)), "south": (grid.ny - 2, slice(1, -1)),
                     "west": (slice(1, -1), 1), "east": (slice(1, -1), grid.nx - 2)}[e]
            line = h[inner]
            k = int(np.argmin(line)) + 1
            r, c = (inner[0], k) if e in ("north", "south") else (k, inner[1])
            out.append({"kind": "pass", "edge": e, "row": int(r), "col": int(c), "elevation_m": float(line[k - 1])})
    return out


def make_external(seed: int, xcfg, grid: Grid, frame: Frame, inflows: list[Inflow], h: np.ndarray) -> dict:
    cands = _candidates(grid, frame, inflows, h)
    ccfg = xcfg.connections
    r = rngmod.stream(seed, "external.connections")
    count = int(r.integers(int(ccfg.count.min), int(ccfg.count.max) + 1))
    weights = np.array([float(ccfg.kind_weights[c["kind"]]) for c in cands])
    chosen = []
    avail = list(range(len(cands)))
    for _ in range(min(count, len(cands))):
        w = weights[avail]
        pick = avail[int(r.choice(len(avail), p=w / w.sum()))]
        avail.remove(pick)
        chosen.append(pick)
    cities = []
    for k in sorted(chosen):
        pop = float(ccfg.city_population.median) * float(np.exp(float(ccfg.city_population.log_sd) * r.standard_normal()))
        cities.append({**cands[k], "population": pop})

    r = rngmod.stream(seed, "external.traits")
    t = xcfg.traits
    return {
        "composition": frame.composition,
        "cities": cities,
        "population_growth_per_year": float(r.uniform(float(t.population_growth_per_year.min), float(t.population_growth_per_year.max))),
        "business_cycle_amplitude": float(r.uniform(float(t.business_cycle_amplitude.min), float(t.business_cycle_amplitude.max))),
    }
