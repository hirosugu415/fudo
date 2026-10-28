"""乱数ストリームの管理（SPEC 原則5）。

ストリームは名前ごとに独立している。新しい名前を追加しても、既存の名前の乱数列は変わらない。
- 世界シード：自然（世界の生成と、年ごとの気候のゆらぎ）
- 実行シード：シミュレーション中の確率的な選択（フェーズ2以降のエージェント）
"""

from __future__ import annotations

import secrets
import zlib

import numpy as np

SEED_BITS = 63


def _key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def _check_seed(seed: int) -> int:
    seed = int(seed)
    if not 0 <= seed < 2**SEED_BITS:
        raise ValueError(f"シードは 0 以上 2^{SEED_BITS} 未満の整数にしてください: {seed}")
    return seed


def stream(world_seed: int, name: str) -> np.random.Generator:
    """世界シードから作る名前付きストリーム。"""
    ss = np.random.SeedSequence(entropy=_check_seed(world_seed), spawn_key=(0, _key(name)))
    return np.random.Generator(np.random.PCG64(ss))


def run_stream(run_seed: int, name: str) -> np.random.Generator:
    """実行シードから作る名前付きストリーム（世界シードのストリームとは別系統）。"""
    ss = np.random.SeedSequence(entropy=_check_seed(run_seed), spawn_key=(1, _key(name)))
    return np.random.Generator(np.random.PCG64(ss))


def derive_run_seed(world_seed: int) -> int:
    """実行シードが指定されないときに、世界シードから決定的に作る。"""
    ss = np.random.SeedSequence(entropy=_check_seed(world_seed), spawn_key=(2, _key("run_seed")))
    hi, lo = (int(x) for x in ss.generate_state(2, dtype=np.uint32))
    return ((hi << 32) | lo) & (2**SEED_BITS - 1)


def random_seed() -> int:
    """シードが空欄のときに使う。OSの乱数から作り、必ず記録・表示する。"""
    return secrets.randbits(SEED_BITS)
