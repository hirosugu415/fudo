"""SPEC 第14章（フェーズ2）：誰も配置せずに、次の現象が起きるか。

- 肥沃な平地に農村が広がる
- 川の合流点や渡し場、峠のふもとに市場町ができる
- 大きい町ほど数が少なく、小さい町ほど数が多い
- 都市の中心ほど地価が高く、離れるほど下がる

100年のウォームアップを実行し、1870年1月の状態を調べる。
"""

import numpy as np
import pytest

from fudo.analysis import (
    load_analysis_config,
    market_town_features,
    near_mask,
    rank_size_slope,
    settlements,
    spearman,
)
from fudo.config import load_behavior_config
from fudo.io import store
from fudo.sim.engine import run_simulation
from fudo.world.generate import generate_world, load_world

WORLD_SEED = 2


@pytest.fixture(scope="module")
def society_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("society")
    wdir = generate_world(WORLD_SEED, out_root=out / "worlds", reuse=False, log=lambda *a: None)
    rdir = run_simulation(wdir, out_root=out / "runs", max_months=1, log=lambda *a: None)
    m = store.read_json(rdir / "manifest.json")
    last = store.read_npz(rdir / "months" / f"{m['months_written'] - 1:05d}.npz")
    _, info, arrays = load_world(wdir)
    return arrays, last


def test_population_and_shops_emerge(society_run):
    _, snap = society_run
    assert float(snap["stat_population"]) > 1000
    assert float(snap["stat_shops"]) > 0
    assert float(snap["stat_farmland_ha"]) > 0


def test_villages_spread_on_fertile_plains(society_run):
    arrays, snap = society_run
    acfg = load_analysis_config()
    land = ~arrays["sea"] & ~arrays["lake"]
    gy, gx = np.gradient(arrays["elevation"], 100.0)
    flat = np.exp(-np.hypot(gx, gy) / 0.05)
    score = arrays["soil_fertility"] * flat * (1.0 + arrays["irrigation"])
    farmed = np.isin(snap["land_use"], (1, 2))
    s = score[land]
    f = farmed[land]
    lo, hi = np.quantile(s, [0.25, 0.75])
    top = f[s >= hi].mean()
    bottom = f[s <= lo].mean()
    assert top > float(acfg.fertile_plains.min_ratio) * max(bottom, 1e-3)


def test_market_towns_at_confluences_fords_and_pass_feet(society_run):
    arrays, snap = society_run
    acfg = load_analysis_config()
    beh = load_behavior_config()
    feats = market_town_features(arrays, snap["road"], acfg, float(beh.transport.roads.visible_level),
                                 float(beh.transport.crossing.ford_max_discharge_m3s), 100.0)
    shops = (snap["shops_daily"] + snap["shops_specialty"] + snap["inns"]).astype(np.float64)
    hh = snap["households"].astype(np.float64)
    checked = 0
    for name, feat in feats.items():
        if not feat.any():
            continue
        near = near_mask(feat, float(acfg.market_towns.near_km), 100.0)
        hh_share = hh[near].sum() / hh.sum()
        if hh[near].sum() < float(acfg.market_towns.min_households_near):
            continue
        shop_share = shops[near].sum() / shops.sum()
        assert shop_share > float(acfg.market_towns.min_ratio) * hh_share, (name, shop_share, hh_share)
        checked += 1
    assert checked >= 1


def test_rank_size_rule(society_run):
    _, snap = society_run
    acfg = load_analysis_config()
    sett = settlements(snap["households"], snap["population"], int(acfg.settlement.dense_households))
    assert len(sett) >= int(acfg.rank_size.min_settlements)
    slope = rank_size_slope(sett, int(acfg.rank_size.top_n))
    lo, hi = acfg.rank_size.slope_range
    assert float(lo) <= slope <= float(hi), slope


def test_land_price_falls_with_distance_from_center(society_run):
    _, snap = society_run
    acfg = load_analysis_config()
    sett = settlements(snap["households"], snap["population"], int(acfg.settlement.dense_households))
    center = sett[0]["center"]
    ny, nx = snap["households"].shape
    rr, cc = np.divmod(np.arange(ny * nx), nx)
    r0, c0 = divmod(center, nx)
    d = np.hypot(rr - r0, cc - c0) * 0.1
    sel = (d <= float(acfg.land_price.radius_km)) & (snap["households"].ravel() > 0)
    rho = spearman(d[sel], snap["land_price"].ravel()[sel])
    assert rho < float(acfg.land_price.max_correlation), rho
