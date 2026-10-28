"""隆起と浸食・堆積による地形の生成（erosion.yaml の式を参照）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import kernels
from .. import rng as rngmod
from ..grid import EDGES, Grid
from ..noise import fbm, normalize01
from .frame import Frame
from .inflow import Inflow


@dataclass
class Boundary:
    fixed: np.ndarray     # (n,) 標高を固定するセル（海と平地の端、峡谷の出口）
    outlet: np.ndarray    # (n,) 水が出ていけるセル（海の端・平地の端・峡谷）
    sea_edge: np.ndarray  # (n,) 海の端のセル（海の起点）
    elevation: np.ndarray  # (n,) 固定セルの標高 [m]


@dataclass
class TerrainResult:
    h: np.ndarray               # (ny, nx) 標高 [m]
    alluvium_m: np.ndarray      # (ny, nx) 沖積層の厚さ [m]
    uplift_m_per_yr: np.ndarray  # (ny, nx)
    sea: np.ndarray             # (ny, nx) bool


def make_boundary(grid: Grid, frame: Frame) -> Boundary:
    n = grid.n
    fixed = np.zeros(n, dtype=bool)
    outlet = np.zeros(n, dtype=bool)
    sea_edge = np.zeros(n, dtype=bool)
    elev = np.zeros(n, dtype=np.float64)
    # 海と平地の端は、標高を固定した水の出口。山地の端は水が外へ出ないだけで、標高は固定しない
    # （ほかのマスと同じく隆起・浸食・崩壊で変わる。固定すると端に沿って切り立った壁ができる）。
    # 角のセルは「海 > 平地」の順で上書きする。
    for kind in ("plain", "sea"):
        for e in EDGES:
            if frame.edge_kind[e] != kind:
                continue
            cells = grid.edge_cells(e)
            fixed[cells] = True
            elev[cells] = frame.outer_elevation[e]
            outlet[cells] = True
            if kind == "sea":
                sea_edge[cells] = True
    if frame.gorge is not None:
        cells = grid.edge_cells(frame.gorge["edge"])[frame.gorge["start"]: frame.gorge["start"] + frame.gorge["width"]]
        fixed[cells] = True
        outlet[cells] = True
        elev[cells] = frame.gorge["elevation"]
    return Boundary(fixed=fixed, outlet=outlet, sea_edge=sea_edge, elevation=elev)


def make_uplift(seed: int, ecfg, xcfg, grid: Grid, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """隆起速度 [m/年] と、沈降盆地の範囲（bool）。"""
    u = ecfg.uplift
    r = rngmod.stream(seed, "uplift")
    umax = float(r.uniform(float(u.max_mm_per_yr.min), float(u.max_mm_per_yr.max))) * 1e-3
    low = float(r.uniform(float(u.lowland_fraction.min), float(u.lowland_fraction.max)))
    n = normalize01(fbm(grid, r, float(u.noise_scale_km), int(u.octaves), float(u.persistence)))
    q = float(np.quantile(n, low))
    # 低地より高いところは隆起、低地は縁で0、奥ほど沈降する（土砂がたまる沈降盆地になる）
    up = umax * np.clip((n - q) / max(1.0 - q, 1e-12), 0.0, None)
    ed = xcfg.edges
    for e in EDGES:
        if frame.edge_kind[e] == "mountain":
            # 外側の山が内側に続くように、山地の端に近い隆起域の隆起を強める（低地は持ち上げない）
            d = grid.distance_to_edge_m(e)
            up = up * (1.0 + float(ed.mountain_edge_uplift_boost) * np.exp(-d / (float(ed.mountain_edge_boost_length_km) * 1000.0)))
    depth = np.clip((q - n) / max(q, 1e-12), 0.0, None)
    sub = float(r.uniform(float(u.basin_subsidence_mm_per_yr.min), float(u.basin_subsidence_mm_per_yr.max))) * 1e-3
    # 沈降の速さの上限：沈降でできる空き（体積/年）が、隆起する山地から出る土砂の一定割合を超えないようにする
    # （山地の削られる速さは長期的に隆起の速さと釣り合う）。これを超えると盆地が埋まりきらず湖になる。
    supply = float(up.sum())
    room = float(depth.sum())
    if room > 0.0:
        sub = min(sub, float(u.basin_fill_ratio) * supply / room)
    U = up - sub * depth
    for e in EDGES:
        if frame.edge_kind[e] == "sea":
            d = grid.distance_to_edge_m(e)
            U = U - float(u.coastal_subsidence_mm_per_yr) * 1e-3 * np.exp(-d / (float(u.coastal_subsidence_length_km) * 1000.0))
    return U, depth > 0.0


def run_erosion(
    grid: Grid,
    h_init: np.ndarray,
    U2: np.ndarray,
    erodibility_fn,
    boundary: Boundary,
    inflows: list[Inflow],
    sediment_yield_m_per_yr: float,
    climate_fn,
    ecfg,
    sea_level: float,
    routing_rng: np.random.Generator,
    basin: np.ndarray | None = None,
) -> tuple[TerrainResult, object]:
    """浸食・堆積を計算する。climate_fn(h2d, sea2d) は平年値と相対流出量を返す。

    段階：① total_yr の浸食（海面 sea_level）→ ② 最終氷期の低海面期 → ③ 完新世（海面を戻し、河口の海を土砂で埋める）。
    ①②では、沈む場所（沈降盆地 basin や海岸の沈降域）が海に沈んだ部分（湾）に入った土砂は湾を埋める。
    外洋に面した帯に届いた分は沖へ運ばれる。
    """
    nx, ny, dx = grid.nx, grid.ny, grid.dx
    area = grid.cell_area
    sp = ecfg.stream_power
    num = ecfg.numerics
    it_cfg = ecfg.sediment_iteration
    every = int(ecfg.time.climate_update_every_steps)
    D_land = float(ecfg.hillslope.D_m2_per_yr)
    sc = float(np.tan(np.deg2rad(float(ecfg.hillslope.critical_angle_deg))))
    max_collapse_iter = int(ecfg.hillslope.collapse_max_iterations)
    collapse_tol = float(ecfg.hillslope.collapse_tolerance_m)
    p_route = float(ecfg.routing.slope_exponent)
    eps = float(num.fill_epsilon_m)
    hist = ecfg.sea_level_history

    h = h_init.ravel().astype(np.float64).copy()
    h[boundary.fixed] = boundary.elevation[boundary.fixed]
    U = U2.ravel().astype(np.float64).copy()
    U[boundary.fixed] = 0.0
    thick = np.zeros(grid.n, dtype=np.float64)

    # 外から流れ込む川：外側の集水面積（相対流出量は陸の平均＝1とみなす）と土砂
    Q_ext = np.zeros(grid.n, dtype=np.float64)
    Qs_ext = np.zeros(grid.n, dtype=np.float64)
    for inf in inflows:
        a = inf.area_km2 * 1e6
        Q_ext[inf.cell] += a
        Qs_ext[inf.cell] += a * sediment_yield_m_per_yr

    D_field = np.full(grid.n, D_land)
    free = ~boundary.fixed
    # 湾として埋めてよい海：沈降盆地のうち、外洋に面した帯（海の端から open_sea_band_km）より内側
    near_open_sea = np.zeros(grid.n, dtype=bool)
    band = float(ecfg.uplift.open_sea_band_km) * 1000.0
    for e in EDGES:
        if boundary.sea_edge[grid.edge_cells(e)].any():
            near_open_sea |= grid.distance_to_edge_m(e).ravel() < band
    # 沈む場所（沈降盆地、または海岸の沈降で隆起より沈降が勝つところ）だけを埋めてよい
    sinking = U < 0.0
    if basin is not None:
        sinking |= basin.ravel()
    bay = free & ~near_open_sea & sinking
    bay_trap = float(ecfg.uplift.basin_trap_fraction)
    state = {"Qs_in": np.zeros(grid.n, dtype=np.float64), "p": None}
    lake_tol = float(num.lake_trap_min_depth_m)

    def advance(total_yr: float, dt: float, level: float, trap: float, fillable: np.ndarray, spread_cells: int) -> None:
        # 海の端の固定標高と海底の下限は、海面に合わせて上下させる
        shift = level - sea_level
        h[boundary.sea_edge] = boundary.elevation[boundary.sea_edge] + shift
        floor = float(num.sea_floor_min_m) + shift
        steps = int(round(total_yr / dt))
        for step in range(steps):
            sea = kernels.connected_below(h, nx, ny, level, boundary.sea_edge)
            if step % every == 0:
                _, p2 = climate_fn(h.reshape(ny, nx), sea.reshape(ny, nx))
                state["p"] = p2.ravel()
            seeds = boundary.outlet | sea
            filled = kernels.priority_flood(h, nx, ny, seeds, eps)
            rcv, rlen = kernels.random_receivers(filled, nx, ny, dx, seeds, routing_rng.random(grid.n), p_route)
            order = np.argsort(filled, kind="stable")
            Q = kernels.accumulate(order, rcv, state["p"] * area + Q_ext)
            K = float(sp.K_base_per_yr) * erodibility_fn(thick.reshape(ny, nx)).ravel()
            h_prev = h.copy()
            is_lake = (filled - h > lake_tol) & ~sea & free
            state["Qs_in"] = kernels.erosion_deposition_step(
                h, U, K, Q, Qs_ext, boundary.fixed, sea, rcv, rlen, order, filled,
                dt, float(sp.m), float(sp.G), area, int(it_cfg.max_iterations), float(it_cfg.tolerance_m),
                floor, state["Qs_in"], lake_tol,
            )
            # 湖に入った土砂を湖の中に広げて積もらせる。満杯の湖からあふれた分は下流へ（海に着いた分は海の土砂に加える）
            lake_vol = np.where(is_lake, state["Qs_in"] * dt, 0.0)
            _, overflow = kernels.lake_fill(h, filled, is_lake, sea, rcv, nx, ny, lake_vol, area)
            # 沖積層は川（と湖）の堆積・下刻だけで増減させる（斜面の拡散でずり落ちた土砂は沖積層に数えない）
            thick[free] = np.maximum(thick[free] + (h[free] - h_prev[free] - U[free] * dt), 0.0)
            target = sea & fillable
            if trap > 0.0 and target.any():
                # 河口の海を、届いた土砂の一部で海面の高さまで埋める。
                # 埋められる海から spread_cells より遠い河口の土砂は、そのまま沖へ運ばれる（探索を省く）
                reach = kernels.distance_from(target, nx, ny, dx) <= spread_cells * dx
                vol = np.where(sea & reach, (state["Qs_in"] * dt + overflow) * trap, 0.0)
                thick[:] += kernels.marine_fill(h, sea, fillable, nx, ny, level, vol, area, spread_cells)
            h_before_diff = h.copy()
            kernels.implicit_diffusion(h, nx, ny, D_field, dt, dx, boundary.fixed | sea)
            # 臨界勾配を超えた斜面は崩れる（崩れた土砂はふもとにたまる）
            kernels.collapse_steep_slopes(h, nx, ny, dx, sc, boundary.fixed, max_collapse_iter, collapse_tol)
            # 拡散と崩壊で削られた分は沖積層から減らす（たまった分は崖錐なので沖積層に数えない）
            thick[free] = np.maximum(thick[free] + np.minimum(h[free] - h_before_diff[free], 0.0), 0.0)

    ho = hist.holocene
    spread = max(1, int(round(float(ho.max_spread_km) * 1000.0 / dx)))
    advance(float(ecfg.time.total_yr), float(ecfg.time.dt_yr), sea_level, bay_trap, bay, spread)
    lo = hist.lowstand
    advance(float(lo.duration_yr), float(lo.dt_yr), sea_level + float(lo.sea_level_offset_m), bay_trap, bay, spread)
    advance(float(ho.duration_yr), float(ho.dt_yr), sea_level, float(ho.marine_trap_fraction), free, spread)

    sea = kernels.connected_below(h, nx, ny, sea_level, boundary.sea_edge)
    thick[sea] = 0.0
    normals, _ = climate_fn(h.reshape(ny, nx), sea.reshape(ny, nx))
    res = TerrainResult(
        h=h.reshape(ny, nx),
        alluvium_m=thick.reshape(ny, nx),
        uplift_m_per_yr=U2,
        sea=sea.reshape(ny, nx),
    )
    return res, normals
