"""店の選び方（Huff モデル）：店を1軒増やしたときの売上の差分計算が、全部やり直した計算と一致すること。"""

import numpy as np

from fudo.society import market


def test_marginal_revenue_matches_full_recompute():
    r = np.random.default_rng(0)
    S, C = 7, 50
    times = r.uniform(0.1, 6.0, size=(S, C))
    times[2, :10] = np.inf          # 行けないところもある
    n = np.array([3.0, 1.0, 0.0, 5.0, 2.0, 0.0, 40.0])   # 0 は出店を考えるだけの場所
    frac = r.uniform(0.0, 0.2, size=S)
    hh = r.integers(0, 30, size=C).astype(np.float64)
    spend = r.uniform(0.0, 5.0, size=C)
    args = (24.0, 0.015, 0.3, 0.5)
    sites = np.array([0, 2, 3, 5], dtype=np.int64)
    fast = market.huff_marginal(times, n, frac, hh, spend, *args, sites)
    for j, s in enumerate(sites):
        rev, _ = market.huff(times, n, frac, hh, spend, *args, int(s), 1.0)
        assert np.isclose(fast[j], rev[s], rtol=1e-10, atol=1e-12)


def test_cached_marginal_and_incremental_update():
    """合計を使い回す計算（huff_base → huff_marginal_cached）と、店が増えたときの書き換え（huff_add_shop）が、全部やり直した計算と一致する。"""
    r = np.random.default_rng(1)
    S, C = 9, 80
    times = r.uniform(0.1, 6.0, size=(S, C))
    times[4, 20:40] = np.inf
    n = np.array([3.0, 1.0, 0.0, 5.0, 2.0, 0.0, 40.0, 1.0, 0.0])
    frac = r.uniform(0.0, 0.2, size=S)
    hh = r.integers(0, 30, size=C).astype(np.float64)
    spend = r.uniform(0.0, 5.0, size=C)
    args = (24.0, 0.015, 0.3, 0.5)
    sites = np.arange(S, dtype=np.int64)
    vmax, tot = market.huff_base(times, n, frac, hh, spend, *args)
    for add in (2, 2, 5, 8):   # 店が増えるたびに書き換え、見込みを全部やり直した計算と比べる
        market.huff_add_shop(times, n, frac, hh, spend, *args, vmax, tot, add)
        n[add] += 1.0
        fast = market.huff_marginal_cached(times, n, frac, hh, spend, *args, vmax, tot, sites)
        slow = market.huff_marginal(times, n, frac, hh, spend, *args, sites)
        assert np.allclose(fast, slow, rtol=1e-9, atol=1e-12)


def test_parallel_dijkstra_matches_serial():
    """並列にまとめた経路計算が、1つずつの計算とビット単位で一致する（決定性）。"""
    from fudo import kernels
    r = np.random.default_rng(2)
    nx, ny = 40, 30
    cost = r.uniform(0.1, 2.0, size=nx * ny)
    cost[r.random(nx * ny) < 0.05] = np.inf
    src = np.array([0, 17, 333, 1199, 600], dtype=np.int64)
    for cut in (np.inf, 8.0):
        dist, pred = kernels.dijkstra_many(cost, nx, ny, src, cut)
        for k, s in enumerate(src):
            d, p = kernels.dijkstra(cost, nx, ny, int(s), cut)
            assert np.array_equal(dist[k], d) and np.array_equal(pred[k], p)
