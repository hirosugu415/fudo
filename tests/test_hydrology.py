"""水系の健全性と水量保存。"""

import numpy as np

from fudo.calendar import days_in_month
from fudo.config import load_generator_config
from fudo.grid import Grid
from fudo.nature.water import WaterModel, WaterState
from fudo.world.generate import water_static_from


def test_every_land_cell_drains_without_cycles(world):
    _, info, a = world
    rcv = a["receiver"].astype(np.int64)
    order = a["order"].astype(np.int64)
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    has = rcv >= 0
    # 流れ先は必ず処理順で先（＝循環がない）
    assert np.all(pos[rcv[has]] < pos[has])
    # 流れ先のないセルは、海か水の出口（開いた端）だけ
    sea = a["sea"].ravel()
    outlet = a["outlet"].ravel()
    assert np.all(sea[~has] | outlet[~has])


def test_rivers_run_in_valley_bottoms(world):
    _, info, a = world
    h = a["elevation"]
    q = a["mean_discharge_m3s"]
    ny, nx = h.shape
    qmin = float(load_generator_config().hydrology.river_min_discharge_m3s)
    river = (q > qmin) & ~a["sea"] & ~a["lake"]
    river[0, :] = river[-1, :] = river[:, 0] = river[:, -1] = False
    rr, cc = np.nonzero(river)
    assert rr.size > 50
    neigh = np.stack([h[rr + dr, cc + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc])
    # 川のマスは、まわり8マスの中央値より低い（谷の底を流れている）
    assert np.mean(h[rr, cc] <= np.median(neigh, axis=0)) > 0.9


def test_water_balance_including_evapotranspiration(world):
    _, info, a = world
    gen = load_generator_config()
    grid = Grid(info["grid"]["nx"], info["grid"]["ny"], info["grid"]["cell_size_m"])
    static = water_static_from(a, grid)
    model = WaterModel(static, gen.hydrology, gen.climate)
    state = WaterState(
        a["init_snow"].copy(), a["init_soil"].copy(), a["init_gw"].copy(),
        a["init_ext_snow"].copy(), a["init_ext_soil"].copy(), a["init_ext_gw"].copy(),
    )
    land = static.land
    total_aet = 0.0
    for k in range(24):
        year, month = 1870 + k // 12, k % 12 + 1
        s0_in, s0_ext = state.storage_m3(land, grid.cell_area, static.ext_area_m2)
        _, st = model.step(state, month, float(days_in_month(year, month)), 0.3, 1.1)
        s1_in, s1_ext = state.storage_m3(land, grid.cell_area, static.ext_area_m2)
        # マップ内：降水 + 外からの流入 = 蒸発散 + 流出 + 貯留の変化
        lhs = st["precip_m3"] + st["inflow_m3"]
        rhs = st["aet_m3"] + st["outflow_m3"] + (s1_in - s0_in)
        assert abs(lhs - rhs) <= 1e-9 * max(lhs, 1.0)
        # 外側の集水域：降水 = 蒸発散 + 流出（= マップへの流入） + 貯留の変化 なので、蒸発散は0以上
        ext_aet = st["ext_precip_m3"] - st["inflow_m3"] - (s1_ext - s0_ext)
        assert ext_aet >= -1e-6 * max(st["ext_precip_m3"], 1.0)
        total_aet += st["aet_m3"]
    assert total_aet > 0.0  # 蒸発散が実際に効いている
