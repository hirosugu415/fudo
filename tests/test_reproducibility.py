"""SPEC 第14章：同じ設定・同じシードで2回実行して結果が完全に一致する。"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from fudo.config.loader import default_config_dir

WORKER = Path(__file__).with_name("repro_worker.py")
SEED = 20260426
MONTHS = 24
WARMUP_YEARS = 10   # 時間を抑えるため、設定を複製してウォームアップを短くする（同じ設定どうしで比べる）


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfg") / "config"
    shutil.copytree(default_config_dir(), d)
    sim = yaml.safe_load((d / "simulation.yaml").read_text(encoding="utf-8"))
    sim["time"]["warmup_years"] = WARMUP_YEARS
    (d / "simulation.yaml").write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")
    return d


def _worker(out: Path, seed: int, months: int, config_dir: Path, physics: str = "-", run_seed: int | None = None) -> dict:
    args = [sys.executable, str(WORKER), str(out), str(seed), str(months), str(config_dir), physics]
    if run_seed is not None:
        args.append(str(run_seed))
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env, check=False)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, config_dir):
    a = _worker(tmp_path_factory.mktemp("a"), SEED, MONTHS, config_dir)
    b = _worker(tmp_path_factory.mktemp("b"), SEED, MONTHS, config_dir)
    return a, b


def test_same_seed_same_world_bitwise(two_runs):
    a, b = two_runs
    assert a["world_id"] == b["world_id"]
    assert a["static_hash"] == b["static_hash"]
    assert a["arrays_hash"] == b["arrays_hash"] == a["static_hash"]
    assert a["world_json_sha"] == b["world_json_sha"]


def test_same_seed_same_run_bitwise(two_runs):
    a, b = two_runs
    assert a["run_id"] == b["run_id"]
    assert len(a["month_hashes"]) == WARMUP_YEARS + MONTHS
    assert a["month_hashes"] == b["month_hashes"]
    assert a["month_hashes"] == a["manifest_month_hashes"]


def test_different_seed_different_world(two_runs, tmp_path, config_dir):
    a, _ = two_runs
    c = _worker(tmp_path, SEED + 1, 1, config_dir)
    assert c["world_id"] != a["world_id"]
    assert c["static_hash"] != a["static_hash"]


def test_physics_change_keeps_world(two_runs, tmp_path, config_dir):
    a, _ = two_runs
    alt = tmp_path / "physics_alt.yaml"
    alt.write_text("laws:\n  test_law: {initial: 0.0, limit: 0.5}\n", encoding="utf-8")
    c = _worker(tmp_path, SEED, 1, config_dir, physics=str(alt))
    assert c["world_id"] == a["world_id"]
    assert c["static_hash"] == a["static_hash"]
    assert c["run_id"] != a["run_id"]


def test_run_seed_changes_run_not_world(two_runs, tmp_path, config_dir):
    a, _ = two_runs
    c = _worker(tmp_path, SEED, 1, config_dir, run_seed=12345)
    assert c["world_id"] == a["world_id"]
    assert c["static_hash"] == a["static_hash"]
    assert c["run_id"] != a["run_id"]
