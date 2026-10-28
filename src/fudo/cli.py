"""コマンドライン：fudo generate / run / describe / view"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _world_dir(arg: str) -> Path:
    from .io import store

    p = Path(arg)
    if p.exists():
        return p
    return store.worlds_dir() / arg


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(prog="fudo", description="都市創発シミュレーター")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="世界シードから世界を生成する")
    g.add_argument("--seed", type=int, default=None, help="世界シード（省略するとランダム）")
    g.add_argument("--json", action="store_true", help="結果を JSON で1行出力する（ビューア用）")

    r = sub.add_parser("run", help="世界でシミュレーションを実行する")
    r.add_argument("--world", required=True, help="world_id または世界のディレクトリ")
    r.add_argument("--physics", default=None, help="物理法則の設定ファイル（省略すると config/physics/default.yaml）")
    r.add_argument("--run-seed", type=int, default=None, help="実行シード（省略すると世界シードから決める）")
    r.add_argument("--months", type=int, default=None, help="計算する月数の上限")

    d = sub.add_parser("describe", help="世界の説明を表示する")
    d.add_argument("--world", required=True)

    v = sub.add_parser("view", help="ビューアを起動する")
    v.add_argument("--port", type=int, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        from . import rng
        from .world.generate import generate_world

        seed = args.seed if args.seed is not None else rng.random_seed()
        log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
        wdir = generate_world(seed, log=log)
        if args.json:
            print(json.dumps({"world_id": wdir.name, "seed": seed}))
        else:
            print(f"world_id: {wdir.name}")
        return 0

    if args.cmd == "run":
        from .sim.engine import run_simulation

        rdir = run_simulation(_world_dir(args.world), physics_path=args.physics, run_seed=args.run_seed,
                              max_months=args.months)
        print(f"run_id: {rdir.name}")
        return 0

    if args.cmd == "describe":
        from .io import store

        info = store.read_json(_world_dir(args.world) / "world.json")
        print(f"シード {info['seed']}（world_id {info['world_id']}）")
        print(info["description"]["text"])
        return 0

    if args.cmd == "view":
        from .viewer.server import serve

        serve(port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
