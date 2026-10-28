"""浸食・堆積：河口や盆地の出口付近に、勾配の緩い堆積面ができること。人工的な地形で確かめる。"""

import copy

import numpy as np
import pytest

from fudo import kernels
from fudo.config import Section, load_generator_config
from fudo.grid import Grid
from fudo.world.terrain import Boundary, run_erosion

MIN_THICK = 2.0  # geology.yaml の alluvium.min_thickness_m と同じ値で判定する


def _ecfg(G: float) -> Section:
    raw = copy.deepcopy(load_generator_config().erosion.raw)
    raw["stream_power"]["G"] = G
    raw["time"]["total_yr"] = 1.5e6
    return Section(raw, "erosion")


def _run(grid, U, boundary, G, basin=None):
    ny, nx = grid.ny, grid.nx
    r = np.random.default_rng(0)
    h0 = r.uniform(0.0, 5.0, size=(ny, nx))
    ones = np.ones((ny, nx))
    res, _ = run_erosion(
        grid, h0, U, lambda th: ones, boundary, [], 0.0, lambda h2, s2: (None, ones), _ecfg(G), 0.0,
        np.random.default_rng(1), basin,
    )
    return res


def _flow_slope(grid, res, boundary):
    """流れの方向に沿った勾配（細い谷で谷壁の勾配を拾わないため）。"""
    h = res.h.ravel()
    seeds = boundary.outlet | res.sea.ravel()
    filled = kernels.priority_flood(h, grid.nx, grid.ny, seeds, 1e-4)
    rcv, dist = kernels.d8_receivers(filled, grid.nx, grid.ny, grid.dx, seeds)
    s = np.zeros_like(h)
    ok = rcv >= 0
    s[ok] = np.maximum(h[ok] - h[rcv[ok]], 0.0) / dist[ok]
    return s.reshape(res.h.shape)


def _coast_case():
    """北半分が隆起する山地、南半分が隆起しない低地、南の端が海。"""
    grid = Grid(48, 72, 100.0)
    n = grid.n
    fixed = np.zeros(n, bool)
    for e in ("north", "south", "east", "west"):
        fixed[grid.edge_cells(e)] = True
    outlet = np.zeros(n, bool)
    outlet[grid.edge_cells("south")] = True
    elev = np.where(fixed, 300.0, 0.0)
    elev[grid.edge_cells("south")] = -40.0
    b = Boundary(fixed=fixed, outlet=outlet, sea_edge=outlet.copy(), elevation=elev)
    U = np.zeros((grid.ny, grid.nx))
    U[: grid.ny // 2] = 1.0e-3
    return grid, U, b


def _basin_case():
    """山に囲まれた隆起しない盆地。南の端の中央の峡谷だけが出口。"""
    grid = Grid(64, 64, 100.0)
    n = grid.n
    fixed = np.zeros(n, bool)
    for e in ("north", "south", "east", "west"):
        fixed[grid.edge_cells(e)] = True
    outlet = np.zeros(n, bool)
    gorge = grid.edge_cells("south")[30:34]
    outlet[gorge] = True
    elev = np.where(fixed, 400.0, 0.0)
    elev[gorge] = 0.0
    rr, cc = np.mgrid[0:64, 0:64]
    basin = np.hypot(rr - 34, cc - 32) < 14
    channel = (cc >= 30) & (cc < 34) & (rr > 34)
    U = np.where(basin | channel, 0.0, 1.0e-3)
    b = Boundary(fixed=fixed, outlet=outlet, sea_edge=np.zeros(n, bool), elevation=elev)
    return grid, U, b, basin


@pytest.fixture(scope="module")
def coast():
    grid, U, b = _coast_case()
    return grid, U, b, _run(grid, U, b, 1.0), _run(grid, U, b, 0.0)


def test_alluvial_plain_near_river_mouth(coast):
    grid, U, b, res, _ = coast
    land = ~res.sea
    low = land & (U == 0)
    low[[0, -1], :] = False
    low[:, [0, -1]] = False
    al = (res.alluvium_m > MIN_THICK) & low
    assert al.sum() > 0.1 * low.sum(), "低地に堆積面ができていない"
    s = _flow_slope(grid, res, b)
    mountain = land & (U > 0)
    mountain[[0, -1], :] = False
    mountain[:, [0, -1]] = False
    assert np.median(s[al]) < 0.2 * np.median(s[mountain]), "堆積面の勾配が緩くない"
    # 河口（海に面した陸のうち流量が最大のマス）のすぐ上流は堆積面になっている
    sea_rows = np.flatnonzero(res.sea.any(axis=1))
    coast_row = sea_rows.min() - 1
    near_mouth = al[max(coast_row - 6, 0): coast_row + 1]
    assert near_mouth.any(), "河口付近に堆積面がない"


def test_no_deposition_without_G(coast):
    grid, U, b, with_g, without_g = coast
    low = (~with_g.sea) & (U == 0)
    a1 = ((with_g.alluvium_m > MIN_THICK) & low).sum()
    a0 = ((without_g.alluvium_m > MIN_THICK) & (~without_g.sea) & (U == 0)).sum()
    assert a0 < 0.2 * a1


def test_alluvial_fill_in_basin_near_outlet():
    grid, U, b, basin = _basin_case()
    res = _run(grid, U, b, 1.0)
    inner = basin & ~res.sea
    al = (res.alluvium_m > MIN_THICK) & inner
    assert al.sum() > 0.2 * inner.sum(), "盆地に堆積面ができていない"
    s = _flow_slope(grid, res, b)
    hills = (U > 0)
    hills[[0, -1], :] = False
    hills[:, [0, -1]] = False
    assert np.median(s[al]) < 0.2 * np.median(s[hills])
    # 盆地の出口側（南の縁）にも堆積面がある
    rr, cc = np.nonzero(al)
    assert (rr > 34 + 8).any(), "盆地の出口付近に堆積面がない"


def test_subsiding_basin_fills_with_sediment():
    """沈降する盆地が、まわりの山の土砂で埋まって平野になる（湖として残らない）。"""
    grid, U, b, basin = _basin_case()
    U = np.where(basin, -0.05e-3, U)  # 盆地の中は 0.05 mm/年 で沈む
    res = _run(grid, U, b, 1.0, basin)
    h = res.h.ravel()
    seeds = b.outlet | res.sea.ravel()
    filled = kernels.priority_flood(h, grid.nx, grid.ny, seeds, 1e-4)
    lake = ((filled - h) > 1.0).reshape(res.h.shape)
    inner = basin & ~res.sea
    assert lake[inner].mean() < 0.05, "沈降盆地が湖のまま残っている"
    al = (res.alluvium_m > MIN_THICK) & inner
    assert al.mean() > 0.0 and al.sum() > 0.5 * inner.sum()
