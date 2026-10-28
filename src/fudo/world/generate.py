"""世界の生成。世界シードと生成器の設定から、自然のすべてを順番に作る（SPEC 第4章）。

順番：枠組み → 地質 → 外から流れ込む川 → 隆起と浸食・堆積（気候と交互に計算） → 水系 → 平年値での水収支
→ 鉱物・森林・漁場 → 外部世界 → 説明。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import GENERATOR_VERSION, kernels
from .. import rng as rngmod
from ..config import canonical_hash, load_generator_config
from ..grid import Grid
from ..io import store
from ..nature.water import WaterModel, WaterStatic, spinup
from ..noise import fbm, normalize01
from .climate_normals import compute_normals, relative_runoff, sea_level_temperature
from .describe import describe_world
from .external import make_external
from .frame import make_frame
from .geology import make_bedrock, rock_property
from .inflow import make_inflows
from .resources import make_fishery, make_forest, make_minerals
from .soil import make_irrigation, make_soil
from .terrain import make_boundary, make_uplift, run_erosion


def generator_hash(gen) -> str:
    return canonical_hash(gen.raw)


def world_id_for(seed: int, gen) -> str:
    return canonical_hash({"seed": int(seed), "generator": gen.raw, "version": GENERATOR_VERSION})[:16]


def water_static_from(arrays: dict, grid: Grid) -> WaterStatic:
    n = grid.n
    return WaterStatic(
        T_norm=arrays["T_norm"].reshape(12, n),
        P_norm=arrays["P_norm"].reshape(12, n),
        land=~arrays["sea"].ravel(),
        order=arrays["order"].astype(np.int64),
        rcv=arrays["receiver"].astype(np.int64),
        cell_area=grid.cell_area,
        daylen_h=arrays["daylen_h"],
        ext_cell=arrays["ext_cell"].astype(np.int64),
        ext_area_m2=arrays["ext_area_m2"],
        ext_T_norm=arrays["ext_T_norm"],
        ext_P_norm=arrays["ext_P_norm"],
    )


def generate_world(seed: int, config_dir: Path | None = None, out_root: Path | None = None,
                   log=print, reuse: bool = True) -> Path:
    gen = load_generator_config(config_dir)
    seed = int(seed)
    wid = world_id_for(seed, gen)
    out_root = Path(out_root) if out_root else store.worlds_dir()
    wdir = out_root / wid
    if reuse and (wdir / "manifest.json").exists():
        log(f"生成済みの世界を使います: {wid}")
        return wdir
    t0 = time.perf_counter()
    grid = Grid.from_config(gen.frame)
    sea_level = float(gen.frame.sea_level_m)

    frame = make_frame(seed, gen, grid)
    bedrock = make_bedrock(seed, gen.geology, grid)
    inflows = make_inflows(seed, gen.hydrology, grid, frame)
    boundary = make_boundary(grid, frame)
    U, basin = make_uplift(seed, gen.erosion, gen.external, grid, frame)
    u = gen.erosion.uplift
    r = rngmod.stream(seed, "terrain.initial")
    h_init = sea_level + float(gen.erosion.initial_relief_m) * normalize01(
        fbm(grid, r, float(u.noise_scale_km), int(u.octaves), float(u.persistence))
    )

    ccfg = gen.climate
    min_rel = float(gen.erosion.numerics.min_relative_runoff)

    def climate_fn(h2, sea2):
        nm = compute_normals(grid, h2, sea2, frame, ccfg)
        return nm, relative_runoff(nm, ~sea2, min_rel)

    def erodibility_fn(thick2):
        return rock_property(gen.geology, bedrock, thick2, "erodibility")

    log("浸食・堆積を計算しています…")
    terrain, normals = run_erosion(
        grid, h_init, U, erodibility_fn, boundary, inflows,
        float(gen.hydrology.inflow.sediment_yield_mm_per_yr) * 1e-3, climate_fn, gen.erosion, sea_level,
        rngmod.stream(seed, "erosion.routing"), basin,
    )
    log(f"  地形ができました（{time.perf_counter() - t0:.1f} 秒）")
    sea = terrain.sea
    # 沖積平野の微起伏（自然堤防・後背湿地）。浸食計算の格子では表せない数十cm〜数mの凹凸で、平野の川の流路を決める
    fp = gen.erosion.floodplain
    alluvial = (terrain.alluvium_m > float(gen.geology.alluvium.min_thickness_m)) & ~sea & ~boundary.fixed.reshape(grid.ny, grid.nx)
    micro = float(fp.micro_relief_m) * fbm(
        grid, rngmod.stream(seed, "floodplain.micro_relief"), float(fp.scale_km), int(fp.octaves), float(fp.persistence)
    )
    h = np.where(alluvial, np.maximum(terrain.h + micro, sea_level + float(fp.min_height_above_sea_m)), terrain.h)

    # 水系（変化しない部分）
    hf = h.ravel()
    seeds = boundary.outlet | sea.ravel()
    # 平らな面の傾きをセルごとにばらつかせて、平野の流路が直線にならないようにする
    jitter = rngmod.stream(seed, "hydrology.flat_routing").random(grid.n)
    eps_arr = float(gen.erosion.numerics.fill_epsilon_m) * (1.0 + float(gen.hydrology.flat_routing_jitter) * jitter)
    filled = kernels.priority_flood_jitter(hf, grid.nx, grid.ny, seeds, eps_arr)
    rcv, _ = kernels.d8_receivers(filled, grid.nx, grid.ny, grid.dx, seeds)
    order = np.argsort(filled, kind="stable")
    lakes = ((filled - hf) > float(gen.hydrology.lake_min_depth_m)).reshape(grid.ny, grid.nx) & ~sea
    land = ~sea
    area_acc = kernels.accumulate(order, rcv, np.where(land.ravel(), grid.cell_area, 0.0))

    # 外側の集水域の平年値：気温はその端の外側の海までの距離と平均標高から、降水はマップの陸の平均とみなす
    lapse = float(ccfg.temperature.lapse_rate_c_per_km) / 1000.0
    k = len(inflows)
    ext_T = np.zeros((12, k))
    ext_P = np.zeros((12, k))
    land_mean_P = normals.P[:, land].mean(axis=1)
    for j, inf in enumerate(inflows):
        for mi in range(12):
            ext_T[mi, j] = float(sea_level_temperature(frame.latitude, mi + 1, frame.outer_sea_km[inf.edge], ccfg)) \
                - lapse * inf.mean_elevation_m
            ext_P[mi, j] = land_mean_P[mi]

    arrays: dict[str, np.ndarray] = {
        "elevation": h.astype(np.float64),
        "sea": sea,
        "lake": lakes,
        "filled": filled.reshape(grid.ny, grid.nx),
        "receiver": rcv.astype(np.int32),
        "order": order.astype(np.int32),
        "outlet": boundary.outlet.reshape(grid.ny, grid.nx),
        "drainage_area_m2": area_acc.reshape(grid.ny, grid.nx),
        "bedrock": bedrock,
        "alluvium_m": terrain.alluvium_m.astype(np.float32),
        "uplift_m_per_yr": terrain.uplift_m_per_yr.astype(np.float32),
        "T_norm": normals.T,
        "P_norm": normals.P,
        "PET_norm": normals.PET.astype(np.float32),
        "sea_dist_km": normals.sea_dist_km.astype(np.float32),
        "q_edge": normals.q_edge,
        "daylen_h": normals.daylen_h,
        "ext_cell": np.array([inf.cell for inf in inflows], dtype=np.int64),
        "ext_area_m2": np.array([inf.area_km2 * 1e6 for inf in inflows], dtype=np.float64),
        "ext_T_norm": ext_T,
        "ext_P_norm": ext_P,
    }

    log("平年値の気候で水収支を慣らしています…")
    model = WaterModel(water_static_from(arrays, grid), gen.hydrology, ccfg)
    state, mean_q, min_q = spinup(model, int(gen.hydrology.spinup_years))
    mean_q2 = mean_q.reshape(grid.ny, grid.nx)
    min_q2 = min_q.reshape(grid.ny, grid.nx)
    arrays.update({
        "mean_discharge_m3s": mean_q2,
        "low_discharge_m3s": min_q2,
        "init_snow": state.snow, "init_soil": state.soil, "init_gw": state.gw,
        "init_ext_snow": state.ext_snow, "init_ext_soil": state.ext_soil, "init_ext_gw": state.ext_gw,
    })

    minerals, deposits = make_minerals(seed, gen.minerals, gen.geology, grid, bedrock, terrain.alluvium_m, land)
    for name, ras in minerals.items():
        arrays[f"mineral_{name}_t"] = ras.astype(np.float32)
    ftype, stock_tab, rate_tab = make_forest(gen.forest, normals, land & ~lakes)
    ha = grid.cell_area / 1e4
    arrays["forest_type"] = ftype
    arrays["forest_max_stock_m3"] = np.where(ftype >= 0, stock_tab[np.maximum(ftype, 0)] * ha, 0.0).astype(np.float32)
    arrays["forest_stock_m3"] = arrays["forest_max_stock_m3"].copy()
    arrays["forest_growth_per_yr"] = np.where(ftype >= 0, rate_tab[np.maximum(ftype, 0)], 0.0).astype(np.float32)
    fkind, fcap, frate = make_fishery(gen.fishery, grid, h, sea, lakes, mean_q2)
    arrays["fish_kind"] = fkind
    arrays["fish_capacity_t"] = fcap.astype(np.float32)
    arrays["fish_growth_per_yr"] = frate.astype(np.float32)

    # 土壌と用水（フェーズ2で追加）
    fert, drain = make_soil(seed, gen.soil, gen.geology, grid, h, sea, bedrock, terrain.alluvium_m)
    arrays["soil_fertility"] = fert
    arrays["soil_drainage"] = drain
    arrays["irrigation"] = make_irrigation(gen.soil.irrigation, grid, h, sea | lakes, min_q2)

    external = make_external(seed, gen.external, grid, frame, inflows, h)

    info = {
        "world_id": wid,
        "seed": seed,
        "grid": {"nx": grid.nx, "ny": grid.ny, "cell_size_m": grid.dx},
        "latitude": frame.latitude,
        "edges": {e: {"kind": frame.edge_kind[e], "outer_elevation_m": frame.outer_elevation[e],
                      "outer_sea_km": frame.outer_sea_km[e], "moisture_factor": frame.moisture_factor[e]}
                  for e in frame.edge_kind},
        "gorge": frame.gorge,
        "rain_systems": frame.rain_systems,
        "wind": {"westerly_speed_ms": frame.westerly_speed, "monsoon_speed_ms": frame.monsoon_speed,
                 "monsoon_summer_from_deg": frame.monsoon_summer_from_deg},
        "inflows": [{"edge": i.edge, "cell": i.cell, "area_km2": i.area_km2, "mean_elevation_m": i.mean_elevation_m}
                    for i in inflows],
        "deposits": deposits,
        "external": external,
    }
    info["description"] = describe_world(gen, grid, info, arrays)

    manifest = {
        "world_id": wid,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "generator_hash": generator_hash(gen),
        "arrays_hash": store.arrays_hash(arrays),
        "environment": store.environment_info(),
    }
    wdir.mkdir(parents=True, exist_ok=True)
    store.write_npz(wdir / "static.npz", arrays)
    store.write_json(wdir / "world.json", info)
    store.write_json(wdir / "manifest.json", manifest)  # 最後に書く（これがあれば完成している）
    log(f"世界 {wid} を生成しました（シード {seed}、{time.perf_counter() - t0:.1f} 秒）")
    return wdir


def load_world(wdir: Path) -> tuple[dict, dict, dict]:
    wdir = Path(wdir)
    return (store.read_json(wdir / "manifest.json"), store.read_json(wdir / "world.json"),
            store.read_npz(wdir / "static.npz"))
