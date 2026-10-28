"""地質：岩盤の種類の分布。沖積層は浸食・堆積の結果から後で決める。"""

from __future__ import annotations

import numpy as np

from .. import rng as rngmod
from ..grid import Grid
from ..noise import fbm


def make_bedrock(seed: int, gcfg, grid: Grid) -> np.ndarray:
    """岩石の番号（geology.yaml の rocks の順）の分布。形は (ny, nx)、int8。"""
    p = gcfg.pattern
    rocks = gcfg.rocks
    scores = []
    for rock in rocks:
        r = rngmod.stream(seed, f"geology.{rock.name}")
        n = fbm(grid, r, float(p.noise_scale_km), int(p.octaves), float(p.persistence), float(p.warp_km))
        n = n / (n.std() + 1e-12)
        scores.append(n + np.log(float(rock.abundance)))
    return np.argmax(np.stack(scores), axis=0).astype(np.int8)


def rock_property(gcfg, bedrock: np.ndarray, alluvium_thick: np.ndarray, name: str) -> np.ndarray:
    """岩石ごとの性質を格子に展開する。沖積層が一定より厚いところは沖積層の性質にする。"""
    table = np.array([float(rock[name]) for rock in gcfg.rocks], dtype=np.float64)
    v = table[bedrock]
    al = alluvium_thick > float(gcfg.alluvium.min_thickness_m)
    v = np.where(al, float(gcfg.alluvium[name]), v)
    return v
