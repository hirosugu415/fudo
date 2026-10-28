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


@njit(cache=True)
def huff_base(times, n_shop, freight_frac, hh_count, spend_per_hh, trips, vot, share, alpha):
    """各マスの「全部の場所の魅力の合計」を、最大値 vmax[c] と、exp(v − vmax) の合計 tot[c] の形で返す。
    出店の見込みを何度も計算するとき、この合計を使い回し、店が増えた場所の分だけ差し替える（huff_add_shop）。"""
    S, C = times.shape
    vmax = np.full(C, -1.0e300)
    tot = np.zeros(C)
    v = np.empty(S)
    for c in range(C):
        sp = spend_per_hh[c]
        if sp <= 0.0 or hh_count[c] <= 0.0:
            continue
        m = -1.0e300
        for s in range(S):
            t = times[s, c]
            if n_shop[s] <= 0.0 or not np.isfinite(t):
                v[s] = -1.0e300
                continue
            v[s] = -(2.0 * trips * t * vot + sp * freight_frac[s]) / (share * sp) + alpha * np.log(n_shop[s])
            if v[s] > m:
                m = v[s]
        if m <= -1.0e299:
            continue
        acc = 0.0
        for s in range(S):
            if v[s] > -1.0e299:
                acc += np.exp(v[s] - m)
        vmax[c] = m
        tot[c] = acc
    return vmax, tot


@njit(cache=True)
def _site_value(times, n, freight_frac, sp, trips, vot, share, alpha, s, c):
    t = times[s, c]
    if n <= 0.0 or not np.isfinite(t):
        return -1.0e300
    return -(2.0 * trips * t * vot + sp * freight_frac[s]) / (share * sp) + alpha * np.log(n)


@njit(cache=True)
def huff_marginal_cached(times, n_shop, freight_frac, hh_count, spend_per_hh, trips, vot, share, alpha, vmax, tot, sites):
    """huff_marginal と同じ値を、huff_base の合計を使って、場所の数に比例しない手間で求める。"""
    S, C = times.shape
    K = sites.size
    out = np.zeros(K)
    for c in range(C):
        sp = spend_per_hh[c]
        if sp <= 0.0 or hh_count[c] <= 0.0:
            continue
        total_spend = sp * hh_count[c]
        for j in range(K):
            s = sites[j]
            if not np.isfinite(times[s, c]):
                continue
            nv = _site_value(times, n_shop[s] + 1.0, freight_frac, sp, trips, vot, share, alpha, s, c)
            ov = _site_value(times, n_shop[s], freight_frac, sp, trips, vot, share, alpha, s, c)
            m = nv if nv > vmax[c] else vmax[c]
            new_tot = tot[c] * np.exp(vmax[c] - m) if vmax[c] > -1.0e299 else 0.0
            if ov > -1.0e299:
                new_tot -= np.exp(ov - m)
            new_tot += np.exp(nv - m)
            out[j] += total_spend * np.exp(nv - m) / new_tot
    return out


@njit(cache=True)
def huff_add_shop(times, n_shop, freight_frac, hh_count, spend_per_hh, trips, vot, share, alpha, vmax, tot, site):
    """場所 site の店が1軒増えたときに、huff_base の合計（vmax, tot）をその場で書き換える。n_shop は呼んだ側で増やす。"""
    S, C = times.shape
    for c in range(C):
        sp = spend_per_hh[c]
        if sp <= 0.0 or hh_count[c] <= 0.0 or not np.isfinite(times[site, c]):
            continue
        nv = _site_value(times, n_shop[site] + 1.0, freight_frac, sp, trips, vot, share, alpha, site, c)
        ov = _site_value(times, n_shop[site], freight_frac, sp, trips, vot, share, alpha, site, c)
        m = nv if nv > vmax[c] else vmax[c]
        t = tot[c] * np.exp(vmax[c] - m) if vmax[c] > -1.0e299 else 0.0
        if ov > -1.0e299:
            t -= np.exp(ov - m)
        tot[c] = t + np.exp(nv - m)
        vmax[c] = m
