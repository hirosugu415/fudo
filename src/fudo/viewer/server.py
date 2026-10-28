"""ビューアのサーバー（FastAPI＋WebSocket）。外部には一切通信しない。

ユーザーの操作は2つだけ（SPEC 原則4）：
- POST /api/worlds   「マップを生成」：シード（空ならランダム）から世界を生成する（別プロセス）
- POST /api/runs     「シミュレート開始」：既定の物理法則でシミュレーションを別プロセスで開始する
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import rng as rngmod
from ..config import load_viewer_config, project_root
from ..io import store
from ..sim.engine import default_run_id

STATIC = Path(__file__).parent / "static"

WORLD_FIELDS = {
    "elevation": np.float32,
    "sea": np.uint8,
    "lake": np.uint8,
    "alluvium_m": np.float32,
    "mean_discharge_m3s": np.float32,
    "forest_type": np.int8,
    "bedrock": np.int8,
}

app = FastAPI(title="fudo viewer")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_cache(request, call_next):
    """画面のファイル（index.html・main.js）をブラウザに古いまま使わせない。更新したのに表示が変わらない、を防ぐ。"""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp
_procs: dict[str, subprocess.Popen] = {}
_world_cache: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    seed: int | None = None


class RunRequest(BaseModel):
    world_id: str


def _safe_id(s: str) -> str:
    if not s.isalnum():
        raise HTTPException(400, "不正なIDです")
    return s


def _world_arrays(wid: str) -> dict:
    wid = _safe_id(wid)
    if wid not in _world_cache:
        p = store.worlds_dir() / wid / "static.npz"
        if not p.exists():
            raise HTTPException(404, "世界がありません")
        _world_cache[wid] = store.read_npz(p)
    return _world_cache[wid]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def viewer_config():
    return load_viewer_config().raw


@app.post("/api/worlds")
async def generate(req: GenerateRequest):
    seed = req.seed if req.seed is not None else rngmod.random_seed()
    if not 0 <= seed < 2**rngmod.SEED_BITS:
        raise HTTPException(400, "シードの範囲が不正です")
    # Windows の uvicorn のイベントループでは非同期サブプロセスが使えないことがあるので、スレッドで待つ
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "fudo.cli", "generate", "--seed", str(seed), "--json"],
        cwd=str(project_root()), capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(500, proc.stderr.decode("utf-8", "replace")[-2000:])
    return json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1])


@app.get("/api/worlds/{wid}")
def world_meta(wid: str):
    wid = _safe_id(wid)
    info = store.read_json(store.worlds_dir() / wid / "world.json")
    a = _world_arrays(wid)
    return {
        "world_id": wid,
        "seed": info["seed"],
        "grid": info["grid"],
        "description": info["description"]["text"],
        "inflows": info["inflows"],
        "external": info["external"],
        "receiver": a["receiver"].tolist(),
    }


@app.get("/api/worlds/{wid}/field/{name}")
def world_field(wid: str, name: str):
    if name not in WORLD_FIELDS:
        raise HTTPException(404, "その変数はありません")
    a = _world_arrays(wid)[name]
    return Response(np.ascontiguousarray(a, dtype=WORLD_FIELDS[name]).tobytes(), media_type="application/octet-stream")


def _default_run_id(wid: str) -> str:
    m = store.read_json(store.worlds_dir() / wid / "manifest.json")
    return default_run_id(wid, int(m["seed"]))


@app.post("/api/runs")
def start_run(req: RunRequest):
    wid = _safe_id(req.world_id)
    if not (store.worlds_dir() / wid / "manifest.json").exists():
        raise HTTPException(404, "世界がありません")
    rid = _default_run_id(wid)
    proc = _procs.get(rid)
    if proc is not None and proc.poll() is None:
        return {"run_id": rid, "status": "running"}
    mpath = store.runs_dir() / rid / "manifest.json"
    if mpath.exists() and store.read_json(mpath).get("complete"):
        return {"run_id": rid, "status": "complete"}
    # 異常終了したときに原因が分かるよう、出力をログファイルに残す（実行のディレクトリは開始時に作り直されるので、その外に置く）
    log_path = store.runs_dir() / f"{rid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        _procs[rid] = subprocess.Popen(
            [sys.executable, "-m", "fudo.cli", "run", "--world", wid], cwd=str(project_root()),
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"},
        )
    return {"run_id": rid, "status": "started"}


def _run_status(rid: str) -> dict | None:
    p = store.runs_dir() / _safe_id(rid) / "manifest.json"
    if not p.exists():
        return None
    try:
        m = store.read_json(p)
    except (json.JSONDecodeError, OSError):
        return None
    return {"run_id": rid, "times": m.get("times", []), "months_total": m["months_total"],
            "months_written": m["months_written"], "complete": m["complete"], "fields": m["fields"]}


@app.get("/api/runs/{rid}")
def run_status(rid: str):
    s = _run_status(rid)
    if s is None:
        raise HTTPException(404, "まだ始まっていません")
    return s


def _month(rid: str, k: int) -> dict:
    s = _run_status(rid)
    if s is None or not 0 <= k < s["months_written"]:
        raise HTTPException(404, "まだ計算されていない月です")
    return store.read_npz(store.runs_dir() / rid / "months" / f"{k:05d}.npz")


@app.get("/api/runs/{rid}/month/{k}/meta")
def month_meta(rid: str, k: int):
    z = _month(rid, k)
    return {key: float(v) for key, v in z.items() if v.ndim == 0}


@app.get("/api/runs/{rid}/month/{k}/field/{name}")
def month_field(rid: str, k: int, name: str):
    z = _month(rid, k)
    if name not in z or z[name].ndim != 2:
        raise HTTPException(404, "その変数はありません")
    return Response(np.ascontiguousarray(z[name], dtype=np.float32).tobytes(), media_type="application/octet-stream")


@app.websocket("/ws/runs/{rid}")
async def run_ws(ws: WebSocket, rid: str):
    await ws.accept()
    last = None
    try:
        while True:
            s = _run_status(rid)
            key = None if s is None else (s["months_written"], s["complete"])
            if s is not None and key != last:
                await ws.send_json(s)
                last = key
            if s is not None and s["complete"]:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


def serve(port: int | None = None) -> None:
    import uvicorn

    cfg = load_viewer_config()
    port = int(port or cfg.port)
    print(f"ビューア: http://{cfg.host}:{port}/")
    uvicorn.run(app, host=str(cfg.host), port=port, log_level="warning")
