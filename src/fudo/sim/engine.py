"""シミュレーションの実行。別プロセスで先に進み、スナップショットを書き出し続ける（SPEC 第2章の実行モデル）。

1. ウォームアップ（既定100年、1年ごと）：何もない土地に外から人が来て、江戸期相当の農村・宿場・漁村が「結果として」できる
2. 本番（1870年1月から毎月）：自然の更新 → 世帯・店の判断 → 市場

上から何かを配置する処理はここに書かない（SPEC 原則1）。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .. import rng as rngmod
from ..calendar import days_in_month, from_index, parse_year_month, to_index
from ..config import (
    canonical_hash,
    load_behavior_config,
    load_generator_config,
    load_physics_config,
    load_simulation_config,
)
from ..grid import Grid
from ..io import store
from ..nature.anomaly import make_series
from ..nature.water import WaterModel, WaterState
from ..society.society import Society
from ..world.generate import load_world, water_static_from


def run_id_for(world_id: str, run_seed: int, physics_hash: str, sim_hash: str, behavior_hash: str) -> str:
    return canonical_hash({"world": world_id, "run_seed": int(run_seed), "physics": physics_hash,
                           "simulation": sim_hash, "behavior": behavior_hash})[:16]


def default_run_id(world_id: str, world_seed: int, config_dir: Path | None = None, physics_path: Path | None = None,
                   run_seed: int | None = None) -> str:
    if run_seed is None:
        run_seed = rngmod.derive_run_seed(int(world_seed))
    return run_id_for(world_id, run_seed, canonical_hash(load_physics_config(physics_path).raw),
                      canonical_hash(load_simulation_config(config_dir).raw),
                      canonical_hash(load_behavior_config(config_dir).raw))


def run_simulation(
    world_dir: Path,
    physics_path: Path | None = None,
    run_seed: int | None = None,
    config_dir: Path | None = None,
    out_root: Path | None = None,
    max_months: int | None = None,
    log=print,
) -> Path:
    """世界 world_dir でシミュレーションを実行し、実行のディレクトリを返す。

    max_months は本番で計算する月数の上限（テスト用）。結果の値には影響しないので run_id には入れない。
    """
    manifest_w, info, arrays = load_world(world_dir)
    gen = load_generator_config(config_dir)
    if manifest_w["generator_hash"] != canonical_hash(gen.raw):
        raise ValueError("世界を生成したときの生成器の設定と、今の設定が違います。世界を生成し直してください")
    sim = load_simulation_config(config_dir)
    beh = load_behavior_config(config_dir)
    world_seed = int(manifest_w["seed"])
    if run_seed is None:
        run_seed = rngmod.derive_run_seed(world_seed)
    physics_hash = canonical_hash(load_physics_config(physics_path).raw)
    sim_hash = canonical_hash(sim.raw)
    beh_hash = canonical_hash(beh.raw)
    rid = run_id_for(manifest_w["world_id"], run_seed, physics_hash, sim_hash, beh_hash)
    out_root = Path(out_root) if out_root else store.runs_dir()
    rdir = out_root / rid
    if rdir.exists():
        shutil.rmtree(rdir)
    (rdir / "months").mkdir(parents=True)

    start = to_index(*parse_year_month(sim.time.start))
    end = to_index(*parse_year_month(sim.time.end))
    if max_months is not None:
        end = min(end, start + int(max_months) - 1)
    start_year = from_index(start)[0]
    warm = int(sim.time.warmup_years)
    warm_years = list(range(start_year - warm, start_year))
    times = [[y, 0] for y in warm_years] + [list(from_index(i)) for i in range(start, end + 1)]
    fields = list(sim.output.month_fields)
    sfields = list(sim.output.society_fields)

    manifest = {
        "run_id": rid,
        "world_id": manifest_w["world_id"],
        "world_seed": world_seed,
        "run_seed": int(run_seed),
        "physics_hash": physics_hash,
        "simulation_hash": sim_hash,
        "behavior_hash": beh_hash,
        "warmup_years": warm,
        "start_index": start,
        "end_index": end,
        "times": times,                 # スナップショットごとの [年, 月]（月 0 はウォームアップの1年）
        "months_total": len(times),
        "months_written": 0,
        "fields": fields + sfields,
        "complete": False,
        "environment": store.environment_info(),
        "month_hashes": [],
    }
    store.write_json(rdir / "manifest.json", manifest)

    grid = Grid(int(info["grid"]["nx"]), int(info["grid"]["ny"]), float(info["grid"]["cell_size_m"]))
    model = WaterModel(water_static_from(arrays, grid), gen.hydrology, gen.climate)
    state = WaterState(
        snow=arrays["init_snow"].copy(), soil=arrays["init_soil"].copy(), gw=arrays["init_gw"].copy(),
        ext_snow=arrays["init_ext_snow"].copy(), ext_soil=arrays["init_ext_soil"].copy(),
        ext_gw=arrays["init_ext_gw"].copy(),
    )
    first_year = warm_years[0] if warm_years else start_year
    anomalies = make_series(world_seed, gen.climate.anomaly, from_index(end)[0])
    if first_year < anomalies.origin_year:
        raise ValueError("ウォームアップの開始年が、気候のゆらぎの起点より前です")
    society = Society(arrays, info, grid, beh, gen, int(run_seed))

    t0 = time.perf_counter()
    shape = (grid.ny, grid.nx)
    mean_q = arrays["mean_discharge_m3s"]

    def write(k: int, snap: dict, stats: dict) -> None:
        for f in sfields:
            snap[f] = society_snap[f]
        for name, v in stats.items():
            snap[f"stat_{name}"] = np.array(v)
        store.write_npz(rdir / "months" / f"{k:05d}.npz", snap)
        manifest["months_written"] = k + 1
        manifest["month_hashes"].append(store.arrays_hash(snap))
        # マニフェストは、そのスナップショットを書き終えてから更新する（ビューアはここまでしか進めない）
        store.write_json(rdir / "manifest.json", manifest)

    k = 0
    # ---- ウォームアップ（1年ごと）----
    for year in warm_years:
        dT, pf = anomalies.at(year)
        society.step_year(dT, pf)
        society_snap = society.snapshot()
        snap = {
            "discharge_m3s": (mean_q * pf).astype(np.float32),
            "year": np.array(year, dtype=np.int32),
            "month": np.array(0, dtype=np.int32),
            "temperature_anomaly_c": np.array(dT),
            "precip_factor": np.array(pf),
        }
        write(k, snap, society.stats())
        k += 1
        if (year - first_year) % 10 == 9:
            st = society.stats()
            log(f"  ウォームアップ {year}年：人口 {st['population']:.0f}、世帯 {st['households']:.0f}、店 {st['shops']:.0f}")

    # ---- 本番（毎月）----
    for idx in range(start, end + 1):
        year, month = from_index(idx)
        dT, pf = anomalies.at(year)
        q, wstats = model.step(state, month, float(days_in_month(year, month)), dT, pf)
        society.step_month(month, dT, pf)
        society_snap = society.snapshot()
        values = {
            "snow_mm": state.snow, "discharge_m3s": q, "groundwater_mm": state.gw, "soil_mm": state.soil,
        }
        snap = {f: values[f].reshape(shape).astype(np.float32) for f in fields}
        snap["year"] = np.array(year, dtype=np.int32)
        snap["month"] = np.array(month, dtype=np.int32)
        snap["temperature_anomaly_c"] = np.array(dT)
        snap["precip_factor"] = np.array(pf)
        write(k, snap, {**wstats, **society.stats()})
        k += 1
    manifest["complete"] = True
    manifest["seconds"] = time.perf_counter() - t0
    store.write_json(rdir / "manifest.json", manifest)
    log(f"実行 {rid} が終わりました（ウォームアップ {warm} 年 + {end - start + 1} か月、{manifest['seconds']:.1f} 秒）")
    return rdir
