"""保存と読み込み。書き込みは一時ファイル → リネームで原子的に行う。

npz ファイルそのもの（zip）には書き込み時刻が入るので、再現性は配列の内容のハッシュ（arrays_hash）で確かめる。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import load_storage_config, project_root


def worlds_dir() -> Path:
    return project_root() / load_storage_config().worlds_dir


def runs_dir() -> Path:
    return project_root() / load_storage_config().runs_dir


def arrays_hash(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(a.dtype.str.encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def _replace_retry() -> tuple[int, float]:
    cfg = load_storage_config().replace_retry
    return int(cfg.count), float(cfg.wait_s)


def _replace(tmp: Path, path: Path) -> None:
    """os.replace。Windows では、別のプロセスが読んでいる最中のファイルを置き換えると
    PermissionError になる（ビューアがマニフェストを読んでいるときなど）。読み手はすぐ閉じるので、待って再試行する。"""
    count, wait = _replace_retry()
    for k in range(count):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == count - 1:
                raise
            time.sleep(wait)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    _atomic_write_bytes(path, buf.getvalue())


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(path: Path, obj) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True)
    _atomic_write_bytes(path, text.encode("utf-8"))


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment_info() -> dict:
    import numba

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
    }
