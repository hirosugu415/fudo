"""再現性テスト用：別プロセスで世界を生成して実行し、結果のハッシュを JSON で出力する。

使い方：python repro_worker.py <出力先> <世界シード> <月数> <設定のディレクトリ> [物理法則のパス] [実行シード]
"""

import hashlib
import json
import sys
from pathlib import Path

from fudo.io import store
from fudo.sim.engine import run_simulation
from fudo.world.generate import generate_world


def main() -> None:
    out = Path(sys.argv[1])
    seed = int(sys.argv[2])
    months = int(sys.argv[3])
    config_dir = Path(sys.argv[4])
    physics = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    run_seed = int(sys.argv[6]) if len(sys.argv) > 6 else None
    quiet = lambda *a: None  # noqa: E731
    wdir = generate_world(seed, config_dir=config_dir, out_root=out / "worlds", reuse=False, log=quiet)
    rdir = run_simulation(wdir, physics_path=physics, run_seed=run_seed, config_dir=config_dir,
                          out_root=out / "runs", max_months=months, log=quiet)
    wm = store.read_json(wdir / "manifest.json")
    rm = store.read_json(rdir / "manifest.json")
    # 書き出したファイルを読み直して、配列の内容から改めてハッシュを取る
    static_hash = store.arrays_hash(store.read_npz(wdir / "static.npz"))
    n = rm["months_written"]
    month_hashes = [store.arrays_hash(store.read_npz(rdir / "months" / f"{k:05d}.npz")) for k in range(n)]
    print(json.dumps({
        "world_id": wdir.name,
        "arrays_hash": wm["arrays_hash"],
        "static_hash": static_hash,
        "world_json_sha": hashlib.sha256((wdir / "world.json").read_bytes()).hexdigest(),
        "run_id": rdir.name,
        "month_hashes": month_hashes,
        "manifest_month_hashes": rm["month_hashes"],
    }))


if __name__ == "__main__":
    main()
