"""店の選び方（Huff モデル型のロジット）と売上の集計。

世帯の店の選び方（commerce.yaml の説明を参照）：
    V_s = −(2 × 年間の回数 × 片道の時間 × 時間の価値 + 支出 × 運賃の割合_s) / (logit_share × 支出) + agglomeration × ln(店の数_s)
    P_s = exp(V_s) / Σ exp(V)
世帯の支出 × P_s が、その場所の店の売上になる（同じ場所の店で等分）。
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def huff(times, n_shop, freight_frac, hh_count, spend_per_hh, trips, vot, share, alpha, site, add):
    """times[s, c]：店の場所 s から世帯のいるマス c までの時間 [時間]（inf は行けない）。

    site >= 0 のとき、その場所の店の数を add だけ増やしたとして計算する（出店の見積もり）。
    戻り値：(売上 [円/年]（場所ごと）, 年間の往復の回数 [回/年]（場所 × マス）)
    """
    S, C = times.shape
    revenue = np.zeros(S)
    flows = np.zeros((S, C))
    v = np.empty(S)
    for c in range(C):
        sp = spend_per_hh[c]
        if sp <= 0.0 or hh_count[c] <= 0.0:
            continue
        vmax = -1.0e300
        for s in range(S):
            n = n_shop[s] + (add if s == site else 0.0)
            t = times[s, c]
            if n <= 0.0 or not np.isfinite(t):
                v[s] = -1.0e300
                continue
            cost = 2.0 * trips * t * vot + sp * freight_frac[s]
            v[s] = -cost / (share * sp) + alpha * np.log(n)
            if v[s] > vmax:
                vmax = v[s]
        if vmax <= -1.0e299:
            continue
        tot = 0.0
        for s in range(S):
            if v[s] > -1.0e299:
                v[s] = np.exp(v[s] - vmax)
                tot += v[s]
            else:
                v[s] = 0.0
        total_spend = sp * hh_count[c]
        for s in range(S):
            if v[s] > 0.0:
                p = v[s] / tot
                revenue[s] += total_spend * p
                flows[s, c] = hh_count[c] * trips * p
    return revenue, flows


@njit(cache=True)
def huff_marginal(times, n_shop, freight_frac, hh_count, spend_per_hh, trips, vot, share, alpha, sites):
    """場所 sites[j] の店を1軒増やしたときの、その場所の売上 [円/年] を、ほかの場所は変えずに求める。

    各マスの「全部の場所の魅力の合計」を1回だけ計算し、増やした場所の分だけ差し替える（huff を場所ごとに
    やり直すより速い。結果は huff(…, site, 1) の売上と同じ）。
    """
    S, C = times.shape
    K = sites.size
    out = np.zeros(K)
    v = np.empty(S)
    for c in range(C):
        sp = spend_per_hh[c]
        if sp <= 0.0 or hh_count[c] <= 0.0:
            continue
        vmax = -1.0e300
        for s in range(S):
            t = times[s, c]
            if n_shop[s] <= 0.0 or not np.isfinite(t):
                v[s] = -1.0e300
                continue
            cost = 2.0 * trips * t * vot + sp * freight_frac[s]
            v[s] = -cost / (share * sp) + alpha * np.log(n_shop[s])
            if v[s] > vmax:
                vmax = v[s]
        tot = 0.0
        if vmax > -1.0e299:
            for s in range(S):
                if v[s] > -1.0e299:
                    tot += np.exp(v[s] - vmax)
        total_spend = sp * hh_count[c]
        for j in range(K):
            s = sites[j]
            t = times[s, c]
            if not np.isfinite(t):
                continue
            cost = 2.0 * trips * t * vot + sp * freight_frac[s]
            nv = -cost / (share * sp) + alpha * np.log(n_shop[s] + 1.0)
            m = nv if nv > vmax else vmax
            new_tot = tot * np.exp(vmax - m) if vmax > -1.0e299 else 0.0
            if v[s] > -1.0e299:
                new_tot -= np.exp(v[s] - m)
            new_tot += np.exp(nv - m)
            out[j] += total_spend * np.exp(nv - m) / new_tot
    return out
