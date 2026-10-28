"""勾配ノイズ（Perlin 型）と fBm。外部ライブラリに頼らず NumPy だけで決定的に作る。"""

from __future__ import annotations

import numpy as np

from .grid import Grid


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def gradient_noise(xs: np.ndarray, ys: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """格子単位の座標 xs, ys（0以上）での勾配ノイズ。値はおよそ [-0.7, 0.7]。"""
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0
    lx = int(x0.max()) + 2
    ly = int(y0.max()) + 2
    ang = rng.uniform(0.0, 2.0 * np.pi, size=(ly, lx))
    gx = np.cos(ang)
    gy = np.sin(ang)

    def dot(ix, iy, dx, dy):
        return gx[iy, ix] * dx + gy[iy, ix] * dy

    n00 = dot(x0, y0, fx, fy)
    n10 = dot(x0 + 1, y0, fx - 1.0, fy)
    n01 = dot(x0, y0 + 1, fx, fy - 1.0)
    n11 = dot(x0 + 1, y0 + 1, fx - 1.0, fy - 1.0)
    u = _fade(fx)
    v = _fade(fy)
    a = n00 + u * (n10 - n00)
    b = n01 + u * (n11 - n01)
    return a + v * (b - a)


def fbm(
    grid: Grid,
    rng: np.random.Generator,
    scale_km: float,
    octaves: int,
    persistence: float,
    warp_km: float = 0.0,
) -> np.ndarray:
    """fBm（オクターブを重ねたノイズ）。振幅で正規化し、およそ [-1, 1]。形は (ny, nx)。"""
    x_m = (np.arange(grid.nx, dtype=np.float64) + 0.5) * grid.dx
    y_m = (np.arange(grid.ny, dtype=np.float64) + 0.5) * grid.dx
    X, Y = np.meshgrid(x_m, y_m)
    if warp_km > 0.0:
        wx = fbm(grid, rng, scale_km, octaves, persistence)
        wy = fbm(grid, rng, scale_km, octaves, persistence)
        X = X + wx * warp_km * 1000.0
        Y = Y + wy * warp_km * 1000.0
    # 座標を正にずらす（格子点の添字が負にならないように）
    shift = warp_km * 1000.0 * 2.0
    total = np.zeros_like(X)
    amp = 1.0
    norm = 0.0
    wavelength = scale_km * 1000.0
    for _ in range(int(octaves)):
        off = rng.uniform(0.0, 1.0, size=2)
        xs = (X + shift) / wavelength + off[0]
        ys = (Y + shift) / wavelength + off[1]
        total += amp * gradient_noise(xs, ys, rng)
        norm += amp
        amp *= persistence
        wavelength /= 2.0
    return total / (norm * 0.7)


def normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)
