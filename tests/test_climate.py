"""気候：風上の斜面で降水が多く風下で少ない、標高が高いほど気温が低い。"""

import numpy as np

from fudo.config import load_generator_config
from fudo.grid import EDGES, Grid
from fudo.world.climate_normals import compute_normals
from fudo.world.frame import Frame


def _frame(lat=35.0):
    f = Frame(
        latitude=lat,
        composition={"mountain": 0.3, "plain": 0.6, "island": 0.1},
        edge_kind={e: "plain" for e in EDGES},
        outer_elevation={e: 20.0 for e in EDGES},
        outer_sea_km={e: 50.0 for e in EDGES},
        moisture_factor={e: 1.0 for e in EDGES},
        gorge=None,
        rain_systems={"baiu": 0.0, "typhoon": 0.0},
        westerly_speed=5.0,
        monsoon_speed=0.0,
        monsoon_summer_from_deg=0.0,
    )
    f.wind = [(1.0, 0.0, 5.0)] * 12  # 一年中、西から東へ吹く
    return f


def test_windward_wetter_than_leeward():
    grid = Grid(80, 40, 100.0)
    cc = np.arange(grid.nx)[None, :].repeat(grid.ny, axis=0)
    h = 1200.0 * np.exp(-(((cc - 40) / 8.0) ** 2))  # 南北に延びる尾根
    sea = np.zeros_like(h, dtype=bool)
    nm = compute_normals(grid, h, sea, _frame(), load_generator_config().climate)
    P = nm.P.sum(axis=0)
    windward = P[:, 30:36].mean()
    leeward = P[:, 44:50].mean()
    assert windward > 1.5 * leeward


def test_temperature_decreases_with_elevation():
    grid = Grid(40, 40, 100.0)
    h = np.linspace(0, 2000, grid.nx)[None, :].repeat(grid.ny, axis=0)
    sea = np.zeros_like(h, dtype=bool)
    nm = compute_normals(grid, h, sea, _frame(), load_generator_config().climate)
    T = nm.T.mean(axis=0)
    assert np.all(np.diff(T[20]) < 0)


def test_moisture_from_sea_edge_exceeds_mountain_edge():
    grid = Grid(40, 40, 100.0)
    h = np.full((grid.ny, grid.nx), 50.0)
    sea = np.zeros_like(h, dtype=bool)
    cfg = load_generator_config().climate
    f_sea = _frame()
    f_sea.edge_kind["west"] = "sea"
    f_sea.outer_sea_km["west"] = 0.0
    f_mtn = _frame()
    f_mtn.edge_kind["west"] = "mountain"
    f_mtn.moisture_factor["west"] = 0.6
    p_sea = compute_normals(grid, h, sea, f_sea, cfg).P.sum()
    p_mtn = compute_normals(grid, h, sea, f_mtn, cfg).P.sum()
    assert p_sea > p_mtn
