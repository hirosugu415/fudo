"""SPEC 第14章（フェーズ3）：鉱山の周りに町ができる。

鉱山は誰も置かない。投資家が出資を決めた鉱床にだけ会社が生まれる。
時間を抑えるため（ユーザーの方針：時間のかかるテストはできるだけなくす）、設定を複製してウォームアップを30年に短くし、
本番は1か月だけ回す（シード2。ウォームアップ中に鉱山が開いて育つ）。
「閉山すると人口が減る」のテストは、閉山が数十年後なので、ユーザーの指示でいったん外した（2026-10-28）。
試しの実行では、浅い炭田の周りが約50世帯 → 掘り尽くして閉山した後は数世帯に減った（phase3_plan.md 第9章）。
"""

import shutil

import pytest
import yaml

from fudo.analysis import households_near, load_analysis_config
from fudo.config.loader import default_config_dir
from fudo.io import store
from fudo.sim.engine import run_simulation
from fudo.world.generate import generate_world

WORLD_SEED = 2
WARMUP_YEARS = 30
MONTHS = 1


@pytest.fixture(scope="module")
def mining_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("mining")
    cfg = out / "config"
    shutil.copytree(default_config_dir(), cfg)
    sim = yaml.safe_load((cfg / "simulation.yaml").read_text(encoding="utf-8"))
    sim["time"]["warmup_years"] = WARMUP_YEARS
    (cfg / "simulation.yaml").write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")
    wdir = generate_world(WORLD_SEED, config_dir=cfg, out_root=out / "worlds", reuse=False, log=lambda *a: None)
    rdir = run_simulation(wdir, config_dir=cfg, out_root=out / "runs", max_months=MONTHS, log=lambda *a: None)
    m = store.read_json(rdir / "manifest.json")
    years = [t[0] for t in m["times"]]
    # 年ごとに最初のスナップショット（ウォームアップの年、本番の1月）の世帯数だけを読む
    hh = {}
    for k, y in enumerate(years):
        if y not in hh:
            hh[y] = store.read_npz(rdir / "months" / f"{k:05d}.npz")["households"]
    return m, hh


def _companies(m):
    out = {}
    for e in m["events"]:
        c = out.setdefault(e["company"], {})
        c[e["event"]] = e
    return out


def test_town_grows_around_mine(mining_run):
    m, hh = mining_run
    a = load_analysis_config().mining_town
    last = max(hh)
    best = None
    for c in _companies(m).values():
        o = c.get("mine_opened")
        if o is None:
            continue
        y0 = o["year"]
        y1 = y0 + int(a.growth_years)
        closed = c.get("mine_closed")
        if y1 > last or (closed and closed["year"] < y1):
            continue
        cell = o["row"] * hh[y0].shape[1] + o["col"]
        near0 = households_near(hh[y0], cell, float(a.radius_km), 100.0)
        near1 = households_near(hh[y1], cell, float(a.radius_km), 100.0)
        local = (near1 + 1.0) / (near0 + 1.0)
        total = hh[y1].sum() / max(hh[y0].sum(), 1)
        ratio = local / total
        if best is None or ratio > best[0]:
            best = (ratio, o["label"], y0, near0, near1)
    assert best is not None, "鉱山が開かなかった"
    assert best[0] >= float(a.min_growth_ratio), best

