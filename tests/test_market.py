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
