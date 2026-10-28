"""年ごとの気候のゆらぎは1700年を起点に生成する。開始年を早めても1870年以降の気候は変わらない。"""

import shutil

import numpy as np
import yaml

from fudo.config import load_generator_config
from fudo.io import store
from fudo.nature.anomaly import make_series
from fudo.sim.engine import run_simulation


def test_series_prefix_independent_of_end_year():
    acfg = load_generator_config().climate.anomaly
    a = make_series(99, acfg, 1950)
    b = make_series(99, acfg, 2150)
    n = a.temperature_c.size
    assert np.array_equal(a.temperature_c, b.temperature_c[:n])
    assert np.array_equal(a.precip_factor, b.precip_factor[:n])
    assert a.origin_year == 1700


def test_earlier_start_keeps_climate_after_1870(world_dir, tmp_path):
    from fudo.config.loader import default_config_dir

    cfg2 = tmp_path / "config"
    shutil.copytree(default_config_dir(), cfg2)
    sim = yaml.safe_load((cfg2 / "simulation.yaml").read_text(encoding="utf-8"))
    sim["time"]["warmup_years"] = 2   # 時間を抑えるため短くする
    (cfg2 / "simulation.yaml").write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")
    cfg3 = tmp_path / "config3"
    shutil.copytree(cfg2, cfg3)
    sim["time"]["start"] = "1850-01"  # 開始（とウォームアップ）を20年早めた場合に相当
    (cfg3 / "simulation.yaml").write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")

    quiet = lambda *a: None  # noqa: E731
    r1870 = run_simulation(world_dir, config_dir=cfg2, out_root=tmp_path / "runs", max_months=24, log=quiet)
    r1850 = run_simulation(world_dir, config_dir=cfg3, out_root=tmp_path / "runs", max_months=20 * 12 + 24, log=quiet)
    assert r1870.name != r1850.name
    t70 = store.read_json(r1870 / "manifest.json")["times"]
    t50 = store.read_json(r1850 / "manifest.json")["times"]
    for (y, m) in [tuple(t) for t in t70 if t[1] > 0]:
        a = store.read_npz(r1870 / "months" / f"{t70.index([y, m]):05d}.npz")
        b = store.read_npz(r1850 / "months" / f"{t50.index([y, m]):05d}.npz")
        assert (int(a["year"]), int(a["month"])) == (int(b["year"]), int(b["month"]))
        assert int(a["year"]) >= 1870
        assert float(a["temperature_anomaly_c"]) == float(b["temperature_anomaly_c"])
        assert float(a["precip_factor"]) == float(b["precip_factor"])
