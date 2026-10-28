"""数値計算の中核（Numba）。

決定性のため parallel=False、fastmath=False。計算順序は配列の順序で固定する。
例外は dijkstra_many だけ：出発点ごとに独立した計算を並列にする（出発点どうしで値を足し合わせないので、結果は並列でも同じ）。
配列はすべて平らにしたセル番号（row * nx + col）で扱う。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .grid import NEIGH_DC, NEIGH_DIST, NEIGH_DR

_DR = NEIGH_DR
_DC = NEIGH_DC
_DD = NEIGH_DIST


# ---------------------------------------------------------------------------
# 優先度付きキュー（2分ヒープ）。キーが同じならセル番号の小さい方を先に出す。
# ---------------------------------------------------------------------------
@njit(cache=True)
def _less(ka, ia, kb, ib):
    return ka < kb or (ka == kb and ia < ib)


@njit(cache=True)
def _heap_push(keys, idxs, size, k, i):
    pos = size
    keys[pos] = k
    idxs[pos] = i
    while pos > 0:
        parent = (pos - 1) // 2
        if _less(keys[pos], idxs[pos], keys[parent], idxs[parent]):
            keys[pos], keys[parent] = keys[parent], keys[pos]
            idxs[pos], idxs[parent] = idxs[parent], idxs[pos]
            pos = parent
        else:
            break
    return size + 1


@njit(cache=True)
def _heap_pop(keys, idxs, size):
    k = keys[0]
    i = idxs[0]
    size -= 1
    keys[0] = keys[size]
    idxs[0] = idxs[size]
    pos = 0
    while True:
        left = 2 * pos + 1
        right = left + 1
        best = pos
        if left < size and _less(keys[left], idxs[left], keys[best], idxs[best]):
            best = left
        if right < size and _less(keys[right], idxs[right], keys[best], idxs[best]):
            best = right
        if best == pos:
            break
        keys[pos], keys[best] = keys[best], keys[pos]
        idxs[pos], idxs[best] = idxs[best], idxs[pos]
        pos = best
    return k, i, size


@njit(cache=True)
def priority_flood_jitter(h, nx, ny, seed, eps):
    """priority_flood と同じだが、平らな面に付ける傾き eps をセルごとに変える（eps は配列）。

    平らな面の流向が出口へ向かう直線にならず、不規則に枝分かれする。
    """
    n = nx * ny
    filled = h.copy()
    done = np.zeros(n, dtype=np.bool_)
    keys = np.empty(n, dtype=np.float64)
    idxs = np.empty(n, dtype=np.int64)
    size = 0
    for i in range(n):
        if seed[i]:
            done[i] = True
            size = _heap_push(keys, idxs, size, filled[i], i)
    while size > 0:
        k, i, size = _heap_pop(keys, idxs, size)
        r = i // nx
        c = i - r * nx
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            if done[j]:
                continue
            done[j] = True
            if filled[j] <= k + eps[j]:
                filled[j] = k + eps[j]
            size = _heap_push(keys, idxs, size, filled[j], j)
    return filled


@njit(cache=True)
def priority_flood(h, nx, ny, seed, eps):
    """Priority-Flood+ε（Barnes et al. 2014）。seed（海・開いた境界）から窪地を埋める。

    埋めた後の面では、seed 以外のどのセルにも、それより低い近傍が必ずある。
    """
    n = nx * ny
    filled = h.copy()
    done = np.zeros(n, dtype=np.bool_)
    keys = np.empty(n, dtype=np.float64)
    idxs = np.empty(n, dtype=np.int64)
    size = 0
    for i in range(n):
        if seed[i]:
            done[i] = True
            size = _heap_push(keys, idxs, size, filled[i], i)
    while size > 0:
        k, i, size = _heap_pop(keys, idxs, size)
        r = i // nx
        c = i - r * nx
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            if done[j]:
                continue
            done[j] = True
            if filled[j] <= k + eps:
                filled[j] = k + eps
            size = _heap_push(keys, idxs, size, filled[j], j)
    return filled


@njit(cache=True)
def d8_receivers(filled, nx, ny, dx, outlet):
    """最急降下の流れ先。outlet は流れ先なし（-1）。距離 [m] も返す。"""
    n = nx * ny
    rcv = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, dx, dtype=np.float64)
    for i in range(n):
        if outlet[i]:
            continue
        r = i // nx
        c = i - r * nx
        best = 0.0
        bj = -1
        bd = dx
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            L = _DD[d] * dx
            s = (filled[i] - filled[j]) / L
            if s > best:
                best = s
                bj = j
                bd = L
        rcv[i] = bj
        dist[i] = bd
    return rcv, dist


@njit(cache=True)
def random_receivers(filled, nx, ny, dx, outlet, u, p):
    """下り方向の近傍から、勾配の p 乗に比例する確率で流れ先を1つ選ぶ（u は [0,1) の一様乱数）。

    時間刻みごとに選び直すと、時間平均で多方向流（MFD, Freeman 1991）に近づく。
    D8 の格子の向きの癖（直線的な川、平行な筋）を減らし、平野では流路が移り変わる。
    """
    n = nx * ny
    rcv = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, dx, dtype=np.float64)
    w = np.empty(8, dtype=np.float64)
    js = np.empty(8, dtype=np.int64)
    ls = np.empty(8, dtype=np.float64)
    for i in range(n):
        if outlet[i]:
            continue
        r = i // nx
        c = i - r * nx
        k = 0
        total = 0.0
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            L = _DD[d] * dx
            s = (filled[i] - filled[j]) / L
            if s > 0.0:
                wk = s ** p
                w[k] = wk
                js[k] = j
                ls[k] = L
                total += wk
                k += 1
        if k == 0:
            continue
        target = u[i] * total
        acc = 0.0
        pick = k - 1
        for t in range(k):
            acc += w[t]
            if acc > target:
                pick = t
                break
        rcv[i] = js[pick]
        dist[i] = ls[pick]
    return rcv, dist


@njit(cache=True)
def accumulate(order, rcv, values):
    """上流から下流へ値を足し合わせる。order は流れ先が先に来る順（昇順）。"""
    acc = values.copy()
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        r = rcv[i]
        if r >= 0:
            acc[r] += acc[i]
    return acc


@njit(cache=True)
def connected_below(h, nx, ny, level, start):
    """start から8近傍でつながり、標高が level 未満のセル（海）を返す。start 自身は必ず含む。"""
    n = nx * ny
    out = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    top = 0
    for i in range(n):
        if start[i]:
            out[i] = True
            stack[top] = i
            top += 1
    while top > 0:
        top -= 1
        i = stack[top]
        r = i // nx
        c = i - r * nx
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            if not out[j] and h[j] < level:
                out[j] = True
                stack[top] = j
                top += 1
    return out


@njit(cache=True)
def distance_from(mask, nx, ny, dx):
    """mask のセルからの距離 [m]（8近傍の2回走査による近似。誤差は数%以内）。mask が空なら無限大。"""
    n = nx * ny
    inf = 1.0e30
    d = np.full(n, inf, dtype=np.float64)
    for i in range(n):
        if mask[i]:
            d[i] = 0.0
    s2 = np.sqrt(2.0) * dx
    for r in range(ny):
        for c in range(nx):
            i = r * nx + c
            v = d[i]
            if c > 0 and d[i - 1] + dx < v:
                v = d[i - 1] + dx
            if r > 0:
                if d[i - nx] + dx < v:
                    v = d[i - nx] + dx
                if c > 0 and d[i - nx - 1] + s2 < v:
                    v = d[i - nx - 1] + s2
                if c < nx - 1 and d[i - nx + 1] + s2 < v:
                    v = d[i - nx + 1] + s2
            d[i] = v
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            i = r * nx + c
            v = d[i]
            if c < nx - 1 and d[i + 1] + dx < v:
                v = d[i + 1] + dx
            if r < ny - 1:
                if d[i + nx] + dx < v:
                    v = d[i + nx] + dx
                if c < nx - 1 and d[i + nx + 1] + s2 < v:
                    v = d[i + nx + 1] + s2
                if c > 0 and d[i + nx - 1] + s2 < v:
                    v = d[i + nx - 1] + s2
            d[i] = v
    return d


@njit(cache=True)
def nearest_source(mask, nx, ny, dx):
    """mask のセルまでの距離 [m] と、最も近い mask のセルの番号（なければ -1）。8近傍の2回走査による近似。"""
    n = nx * ny
    inf = 1.0e30
    d = np.full(n, inf, dtype=np.float64)
    src = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if mask[i]:
            d[i] = 0.0
            src[i] = i
    s2 = np.sqrt(2.0) * dx
    for sweep in range(2):
        rows = range(ny) if sweep == 0 else range(ny - 1, -1, -1)
        for r in rows:
            cols = range(nx) if sweep == 0 else range(nx - 1, -1, -1)
            for c in cols:
                i = r * nx + c
                sg = 1 if sweep == 0 else -1
                # 走査の向きで、すでに処理した側の4つの近傍を見る
                for k in range(4):
                    if k == 0:
                        rr, cc, w = r, c - sg, dx
                    elif k == 1:
                        rr, cc, w = r - sg, c, dx
                    elif k == 2:
                        rr, cc, w = r - sg, c - sg, s2
                    else:
                        rr, cc, w = r - sg, c + sg, s2
                    if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                        continue
                    j = rr * nx + cc
                    if d[j] + w < d[i]:
                        d[i] = d[j] + w
                        src[i] = src[j]
    return d, src


# ---------------------------------------------------------------------------
# 最小コスト経路（ダイクストラ法、8近傍）
# ---------------------------------------------------------------------------
@njit(cache=True)
def dijkstra(cost, nx, ny, source, cutoff):
    """セルを1つ横切るコスト cost から、source からの最小コストと、経路をたどるための前のセルを求める。

    隣のセルへ移るコスト = (cost[i] + cost[j]) / 2 × (斜めなら √2)。cutoff を超えるところは探索しない（距離は無限大）。
    コストが同じなら、セル番号の小さい方を先に処理する（決定性）。
    """
    n = nx * ny
    dist = np.full(n, np.inf, dtype=np.float64)
    pred = np.full(n, -1, dtype=np.int32)
    done = np.zeros(n, dtype=np.bool_)
    keys = np.empty(8 * n + 1, dtype=np.float64)
    idxs = np.empty(8 * n + 1, dtype=np.int64)
    dist[source] = 0.0
    size = _heap_push(keys, idxs, 0, 0.0, source)
    while size > 0:
        k, i, size = _heap_pop(keys, idxs, size)
        if done[i]:
            continue
        done[i] = True
        if k > cutoff:
            break
        r = i // nx
        c = i - r * nx
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            if done[j]:
                continue
            nd = k + 0.5 * (cost[i] + cost[j]) * _DD[d]
            if nd < dist[j]:
                dist[j] = nd
                pred[j] = i
                size = _heap_push(keys, idxs, size, nd, j)
    for i in range(n):
        if dist[i] > cutoff:
            dist[i] = np.inf
            pred[i] = -1
    return dist, pred


@njit(cache=True, parallel=True)
def dijkstra_many(cost, nx, ny, sources, cutoff):
    """複数の出発点 sources からの dijkstra を並列に計算する。戻り値：(dist[k, セル], pred[k, セル])。

    出発点ごとの計算は互いに独立で、書き込む行も別なので、並列でも結果は dijkstra を1つずつ呼んだのと同じ。
    """
    K = sources.size
    n = nx * ny
    dist = np.empty((K, n), dtype=np.float64)
    pred = np.empty((K, n), dtype=np.int32)
    for k in prange(K):
        d, p = dijkstra(cost, nx, ny, sources[k], cutoff)
        dist[k] = d
        pred[k] = p
    return dist, pred


@njit(cache=True)
def add_path_traffic(pred, origins, amounts, traffic):
    """各出発点から、pred をたどって源（pred = -1）まで、通行量 amounts を traffic に足す。"""
    n = pred.size
    for o in range(origins.size):
        i = origins[o]
        a = amounts[o]
        if a <= 0.0:
            continue
        guard = 0
        while i >= 0 and guard < n:
            traffic[i] += a
            i = pred[i]
            guard += 1


# ---------------------------------------------------------------------------
# 浸食と堆積（Yuan et al. 2019、n = 1）
# ---------------------------------------------------------------------------
@njit(cache=True)
def erosion_deposition_step(
    h, U, K, Q, Qs_ext, fixed, sea, rcv, rlen, order, filled,
    dt, m, G, area, max_iter, tol, sea_floor_min, Qs_in_init, lake_tol,
):
    """1時間刻みの河川の下刻と堆積を陰解法で解く。h を書き換え、収束した流入土砂量 Qs_in [m³/年] を返す。

    - 一般のセル：h' = (h + dt·(U + G·Qs_in/Q) + F·h'_r) / (1 + F)、F = K·Q^m·dt / 流路長
      堆積の速さは、流れ込む土砂を全部積もらせる速さ（Qs_in/面積）を上限とする。
    - 窪地（湖、深さ lake_tol 超）のセル：下刻しない。流れ込んだ土砂は下流へ流さず、lake_fill で湖の中に広げる。
    - 海：隆起（沈降）だけ。流れ込んだ土砂は沖へ運ばれるとみなす。
    - 固定セル（海や開いた陸の境界、陸の壁）：変えない。
    Qs_in は流れ先の新しい標高に依存しないが上流の新しい標高に依存するため、Gauss-Seidel 型に反復する。
    """
    n = h.size
    h0 = h.copy()
    hn = h.copy()
    Qs_in = Qs_in_init.copy()
    Qs_new = np.zeros(n, dtype=np.float64)
    for it in range(max_iter):
        maxdiff = 0.0
        for k in range(n):
            i = order[k]
            if fixed[i]:
                continue
            base = h0[i] + U[i] * dt
            if sea[i]:
                v = base
                if v < sea_floor_min:
                    v = sea_floor_min
            elif filled[i] - h0[i] > lake_tol:
                # 湖：下刻しない。流れ込んだ土砂は湖の中に広げて積もらせる（lake_fill で計算する）
                v = base
            else:
                r = rcv[i]
                dep = 0.0
                if Q[i] > 0.0:
                    dep = G * Qs_in[i] / Q[i]
                    cap = Qs_in[i] / area
                    if dep > cap:
                        dep = cap
                if r >= 0:
                    F = K[i] * Q[i] ** m * dt / rlen[i]
                    v = (base + dt * dep + F * hn[r]) / (1.0 + F)
                else:
                    v = base + dt * dep
            dv = abs(v - hn[i])
            if dv > maxdiff:
                maxdiff = dv
            hn[i] = v
        for i in range(n):
            Qs_new[i] = 0.0
        for k in range(n - 1, -1, -1):
            i = order[k]
            if fixed[i] or sea[i]:
                # 固定セルのうち陸の壁は土砂を下流へ流す（外から入る川の入口など）
                if fixed[i] and not sea[i] and rcv[i] >= 0:
                    Qs_new[rcv[i]] += Qs_new[i] + Qs_ext[i]
                continue
            if filled[i] - h0[i] > lake_tol:
                # 湖のセルは土砂の受け皿（下流へは流さない）。入ってきた量は Qs_in として返す
                Qs_new[i] += Qs_ext[i]
                continue
            out = Qs_new[i] + Qs_ext[i] + (h0[i] + U[i] * dt - hn[i]) * area / dt
            if out < 0.0:
                out = 0.0
            r = rcv[i]
            if r >= 0:
                Qs_new[r] += out
        for i in range(n):
            Qs_in[i] = Qs_new[i]
        if it > 0 and maxdiff < tol:
            break
    for i in range(n):
        h[i] = hn[i]
    return Qs_in


@njit(cache=True)
def marine_fill(h, sea, fillable, nx, ny, level, volume, area, max_radius):
    """海に届いた土砂で、河口（volume > 0 の海のセル）から近い順に海を海面の高さまで埋める（三角州・溺れ谷・湾の埋め立て）。

    volume [m³] は海のセルごとの、そこに流れ込んだ土砂の量。fillable のセルだけを埋め、それ以外の海は通り抜ける。
    max_radius（マス）の中で埋めきれなかった分は沖へ運ばれる。h を書き換え、各セルに積もった厚さ [m] を返す。
    """
    n = nx * ny
    dep = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int64)  # 何番目の河口で訪れたか（0 は未訪問）
    keys = np.empty(n, dtype=np.float64)
    idxs = np.empty(n, dtype=np.int64)
    r2max = float(max_radius * max_radius)
    tag = 0
    for s in range(n):
        v = volume[s]
        if v <= 0.0 or not sea[s]:
            continue
        tag += 1
        sr = s // nx
        sc = s - sr * nx
        size = _heap_push(keys, idxs, 0, 0.0, s)
        seen[s] = tag
        while size > 0 and v > 0.0:
            k, i, size = _heap_pop(keys, idxs, size)
            if fillable[i]:
                room = (level - h[i]) * area
                if room > 0.0:
                    take = room if room < v else v
                    h[i] += take / area
                    dep[i] += take / area
                    v -= take
            r = i // nx
            c = i - r * nx
            for d in range(8):
                rr = r + _DR[d]
                cc = c + _DC[d]
                if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                    continue
                j = rr * nx + cc
                if seen[j] == tag or not sea[j]:
                    continue
                seen[j] = tag
                dist2 = float((rr - sr) * (rr - sr) + (cc - sc) * (cc - sc))
                if dist2 > r2max:
                    continue
                size = _heap_push(keys, idxs, size, dist2, j)
    return dep


@njit(cache=True)
def _fill_from(h, filled, is_lake, nx, ny, start, v, area, dep, seen, tag, keys, idxs):
    """start から近い順に、つながった湖のセルを水面（filled）の高さまで埋める。残った量を返す。"""
    sr = start // nx
    sc = start - sr * nx
    size = _heap_push(keys, idxs, 0, 0.0, start)
    seen[start] = tag
    while size > 0 and v > 0.0:
        k, i, size = _heap_pop(keys, idxs, size)
        room = (filled[i] - h[i]) * area
        if room > 0.0:
            take = room if room < v else v
            h[i] += take / area
            dep[i] += take / area
            v -= take
        r = i // nx
        c = i - r * nx
        for d in range(8):
            rr = r + _DR[d]
            cc = c + _DC[d]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            j = rr * nx + cc
            if seen[j] == tag or not is_lake[j]:
                continue
            seen[j] = tag
            size = _heap_push(keys, idxs, size, float((rr - sr) * (rr - sr) + (cc - sc) * (cc - sc)), j)
    return v


@njit(cache=True)
def lake_fill(h, filled, is_lake, sea, rcv, nx, ny, volume, area):
    """湖に入った土砂を、入口から近い順に湖の水面（filled）の高さまで積もらせる。

    湖が満杯になって余った土砂は、同じ時間刻みのうちに流れ先をたどって下流へ運ぶ。途中に別の湖があればそれを埋め、
    海に着いたらそのセルに、マップの外に出たら出口のセルに渡す。h を書き換え、(積もった厚さ [m], 海・出口に渡した土砂 [m³]) を返す。
    """
    n = nx * ny
    dep = np.zeros(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int64)
    keys = np.empty(n, dtype=np.float64)
    idxs = np.empty(n, dtype=np.int64)
    tag = 0
    for s in range(n):
        v = volume[s]
        if v <= 0.0 or not is_lake[s]:
            continue
        j = s
        guard = 0
        while v > 0.0 and guard < n:
            guard += 1
            if is_lake[j]:
                tag += 1
                v = _fill_from(h, filled, is_lake, nx, ny, j, v, area, dep, seen, tag, keys, idxs)
                if v <= 0.0:
                    break
                # 満杯になったこの湖を抜けるまで流れ先をたどる
                while is_lake[j] and rcv[j] >= 0 and guard < n:
                    j = rcv[j]
                    guard += 1
                if is_lake[j]:
                    break
            if sea[j] or rcv[j] < 0:
                break
            j = rcv[j]
        out[j] += v
    return dep, out


@njit(cache=True)
def collapse_steep_slopes(h, nx, ny, dx, sc, fixed, max_iter, tol):
    """隣のマスとの勾配が臨界勾配 sc（tan θc）を超えたら、超えた分を低い側へ崩す（質量は保存する）。

    8近傍の組ごとに (h_i − h_j)/L − sc > 0 なら、その半分ずつを高い側から低い側へ移す。
    低い側が固定セル（海・平地の端）なら、高い側を臨界勾配まで下げ、崩れた土砂は外へ出る。
    すべての組が臨界勾配以下になるか max_iter 回まで繰り返す。崩れて下がった量 [m] を返す。
    """
    n = nx * ny
    lost = np.zeros(n, dtype=np.float64)
    for it in range(max_iter):
        worst = 0.0
        for i in range(n):
            if fixed[i]:
                continue
            r = i // nx
            c = i - r * nx
            for d in range(8):
                rr = r + _DR[d]
                cc = c + _DC[d]
                if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                    continue
                j = rr * nx + cc
                excess = h[i] - h[j] - sc * _DD[d] * dx
                if excess <= tol:
                    continue
                if excess > worst:
                    worst = excess
                if fixed[j]:
                    move = excess
                else:
                    move = 0.5 * excess
                    h[j] += move
                h[i] -= move
                lost[i] += move
        if worst <= tol:
            break
    return lost


@njit(cache=True)
def _thomas(a, b, c, d, x, m):
    """三重対角の連立方程式を解く（a: 下、b: 対角、c: 上）。"""
    cp = np.empty(m)
    dp = np.empty(m)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, m):
        den = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / den
        dp[k] = (d[k] - a[k] * dp[k - 1]) / den
    x[m - 1] = dp[m - 1]
    for k in range(m - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]


@njit(cache=True)
def _face(Da, Db):
    if Da <= 0.0 or Db <= 0.0:
        return 0.0
    return 2.0 * Da * Db / (Da + Db)


@njit(cache=True)
def implicit_diffusion(h, nx, ny, D, dt, dx, fixed):
    """拡散 ∂h/∂t = ∇·(D∇h) を陰解法で解く（行方向・列方向に分けた後退オイラー。無条件に安定）。

    D はセルごとの拡散係数 [m²/年]、セルの境目では調和平均を使う。
    マップの端は流束0、fixed のセルは値を変えない（そこへ流れ出た分は外へ出る）。
    """
    r = dt / (dx * dx)
    m = max(nx, ny)
    a = np.empty(m)
    b = np.empty(m)
    c = np.empty(m)
    d = np.empty(m)
    x = np.empty(m)
    for row in range(ny):
        for col in range(nx):
            i = row * nx + col
            d[col] = h[i]
            if fixed[i]:
                a[col] = 0.0
                b[col] = 1.0
                c[col] = 0.0
                continue
            kw = _face(D[i], D[i - 1]) if col > 0 else 0.0
            ke = _face(D[i], D[i + 1]) if col < nx - 1 else 0.0
            a[col] = -r * kw
            c[col] = -r * ke
            b[col] = 1.0 + r * (kw + ke)
        _thomas(a, b, c, d, x, nx)
        for col in range(nx):
            h[row * nx + col] = x[col]
    for col in range(nx):
        for row in range(ny):
            i = row * nx + col
            d[row] = h[i]
            if fixed[i]:
                a[row] = 0.0
                b[row] = 1.0
                c[row] = 0.0
                continue
            kn = _face(D[i], D[i - nx]) if row > 0 else 0.0
            ks = _face(D[i], D[i + nx]) if row < ny - 1 else 0.0
            a[row] = -r * kn
            c[row] = -r * ks
            b[row] = 1.0 + r * (kn + ks)
        _thomas(a, b, c, d, x, ny)
        for row in range(ny):
            h[row * nx + col] = x[row]


# ---------------------------------------------------------------------------
# 気候：水蒸気の移流と降水
# ---------------------------------------------------------------------------
@njit(cache=True)
def upwind_precipitation(
    nx, ny, dx, wx, wy, speed, slope_w, sea, q_edge, q_sea,
    rainout_len, oro_gain, oro_max, lee_supp, lee_floor, sea_relax_len, seconds,
):
    """風に沿って上流から水蒸気量 q [mm] を運び、降水 [mm/期間] を返す。

    wx, wy：風の向きの単位ベクトル（列方向=東、行方向=南）。
    q_edge：[北, 東, 南, 西] の端から入る空気の水蒸気量 [mm]。
    定常の風上差分：q·(|wx|+|wy|) + q·φ·dx/L (+ 海の上の補給) = |wx|·q_x + |wy|·q_y
    """
    n = nx * ny
    q = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=np.float64)
    ax = abs(wx)
    ay = abs(wy)
    sx = 1 if wx > 0 else -1
    sy = 1 if wy > 0 else -1
    # 風上から順に処理する：列は sx の向き、行は sy の向きに進めれば、上流の値が先に決まる
    for a in range(ny):
        r = a if sy > 0 else ny - 1 - a
        for b in range(nx):
            c = b if sx > 0 else nx - 1 - b
            i = r * nx + c
            cu = c - sx
            if cu < 0:
                qx = q_edge[3]
            elif cu >= nx:
                qx = q_edge[1]
            else:
                qx = q[r * nx + cu]
            ru = r - sy
            if ru < 0:
                qy = q_edge[0]
            elif ru >= ny:
                qy = q_edge[2]
            else:
                qy = q[ru * nx + c]
            s = slope_w[i]
            if sea[i]:
                phi = 1.0
            elif s >= 0.0:
                phi = 1.0 + oro_gain * s
                if phi > oro_max:
                    phi = oro_max
            else:
                phi = 1.0 - lee_supp * (-s)
                if phi < lee_floor:
                    phi = lee_floor
            num = ax * qx + ay * qy
            den = ax + ay + phi * dx / rainout_len
            if sea[i]:
                num += q_sea * dx / sea_relax_len
                den += dx / sea_relax_len
            qi = num / den
            q[i] = qi
            P[i] = qi * speed * phi / rainout_len * seconds
    return P


# ---------------------------------------------------------------------------
# 月の水収支（積雪・蒸発散・土壌・地下水・流量）
# ---------------------------------------------------------------------------
@njit(cache=True)
def hamon_pet_mm(T, daylen_h, days, coef, hamon_const, es_a, es_b, es_c, vd_factor, vd_offset):
    """Hamon 法の可能蒸発散量 [mm/期間]。"""
    es = es_a * np.exp(es_b * T / (T + es_c))
    rho = vd_factor * es / (T + vd_offset)
    return coef * hamon_const * (daylen_h / 12.0) * rho * days


@njit(cache=True)
def water_month(
    T, P, land, order, rcv, area,
    snow, soil, gw,
    days, daylen_h,
    t_snow, t_melt, ddf, soil_cap, f_inf, gw_res_days,
    pet_coef, hamon_const, es_a, es_b, es_c, vd_factor, vd_offset,
    ext_cell, ext_volume,
):
    """1か月の水収支。snow, soil, gw [mm] を書き換え、各セルを通る流量 [m³/月] と集計値を返す。

    集計値 diag = [降水, 蒸発散, 流出（海・マップ外へ）, 外からの流入] [m³/月]
    陸のセルの流れ先が海（land でない）か -1 なら、そこで流出とする。
    """
    n = T.size
    local = np.zeros(n, dtype=np.float64)
    k_gw = 1.0 - np.exp(-days / gw_res_days)
    p_sum = 0.0
    aet_sum = 0.0
    for i in range(n):
        if not land[i]:
            continue
        t = T[i]
        p = P[i]
        p_sum += p
        if t < t_snow:
            snow[i] += p
            rain = 0.0
        else:
            rain = p
        melt = 0.0
        if t > t_melt:
            melt = ddf * (t - t_melt) * days
            if melt > snow[i]:
                melt = snow[i]
        snow[i] -= melt
        s = soil[i] + rain + melt
        pet = hamon_pet_mm(t, daylen_h, days, pet_coef, hamon_const, es_a, es_b, es_c, vd_factor, vd_offset)
        aet = pet if pet < s else s
        if aet < 0.0:
            aet = 0.0
        s -= aet
        aet_sum += aet
        excess = 0.0
        if s > soil_cap:
            excess = s - soil_cap
            s = soil_cap
        soil[i] = s
        recharge = excess * f_inf
        g = gw[i] + recharge
        base = g * k_gw
        gw[i] = g - base
        local[i] = (excess - recharge + base) * 1.0e-3 * area
    ext_sum = 0.0
    out_sum = 0.0
    for e in range(ext_cell.size):
        ext_sum += ext_volume[e]
        if land[ext_cell[e]]:
            local[ext_cell[e]] += ext_volume[e]
        else:
            # 流入点が海になっている（沈降で沈んだ）ときは、そのまま海への流出とする
            out_sum += ext_volume[e]
    flow = local.copy()
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        if not land[i]:
            continue
        r = rcv[i]
        if r >= 0 and land[r]:
            flow[r] += flow[i]
        else:
            out_sum += flow[i]
    diag = np.empty(4, dtype=np.float64)
    diag[0] = p_sum * 1.0e-3 * area
    diag[1] = aet_sum * 1.0e-3 * area
    diag[2] = out_sum
    diag[3] = ext_sum
    return flow, diag


@njit(cache=True)
def local_maxima(h, nx, ny, radius, valid):
    """半径 radius（マス）の中で最も高いセル。同じ高さならセル番号の小さい方。"""
    n = nx * ny
    out = np.zeros(n, dtype=np.bool_)
    r2 = radius * radius
    for i in range(n):
        if not valid[i]:
            continue
        r = i // nx
        c = i - r * nx
        hi = h[i]
        ok = True
        for dr in range(-radius, radius + 1):
            rr = r + dr
            if rr < 0 or rr >= ny:
                continue
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > r2:
                    continue
                cc = c + dc
                if cc < 0 or cc >= nx:
                    continue
                j = rr * nx + cc
                if j == i:
                    continue
                if h[j] > hi or (h[j] == hi and j < i):
                    ok = False
                    break
            if not ok:
                break
        out[i] = ok
    return out


# ---------------------------------------------------------------------------
# 農家が田畑を手に入れる（society._acquire_land と同じ規則を、多くの農家について順にまとめて行う）
# ---------------------------------------------------------------------------
@njit(cache=True)
def acquire_land_many(want, home, labor, have, area, pot, labor_per_ha, reclaim_py, crop, arable,
                      use, owner, cultivator, reclaim_left, nx, ny, r, radius_km, dx, loss_per_km, reclaim_max, wild):
    """want の順に、各農家 h が働き手 labor[h] で耕せる面積まで、家の近くの田畑を手に入れる。
    優先順：持ち主のいない田畑 → 借りられる田畑 → 未開の土地の開墾（1年に reclaim_max ha まで）。
    同じ種類の中では、見込みの収穫が多く家に近い順（同じなら窓の中の順番）。have[h] はその場で増やす。"""
    w = 2 * r + 1
    cells = np.empty(w * w, dtype=np.int64)
    score = np.empty(w * w)
    grp = np.empty(w * w, dtype=np.int64)
    for q in range(want.size):
        h = want[q]
        L = labor[h]
        if have[h] >= L:
            continue
        hr = home[h] // nx
        hc = home[h] - hr * nx
        m = 0
        for rr in range(max(hr - r, 0), min(hr + r + 1, ny)):
            for cc in range(max(hc - r, 0), min(hc + r + 1, nx)):
                c = rr * nx + cc
                d = np.hypot(rr - hr, cc - hc) * dx / 1000.0
                if d > radius_km or area[c] <= 0.0:
                    continue
                g = -1
                if use[c] > 0 and owner[c] < 0 and cultivator[c] < 0:
                    g = 0
                elif use[c] > 0 and owner[c] >= 0 and cultivator[c] < 0:
                    g = 1
                elif use[c] == wild and arable[c] and owner[c] < 0:
                    g = 2
                if g < 0:
                    continue
                cells[m] = c
                score[m] = pot[c] / (1.0 + loss_per_km * d)
                grp[m] = g
                m += 1
        reclaimed = 0.0
        done = False
        for gi in range(3):
            if done:
                break
            k = 0
            idx = np.empty(m, dtype=np.int64)
            key = np.empty(m)
            for j in range(m):
                if grp[j] == gi:
                    idx[k] = j
                    key[k] = -score[j]
                    k += 1
            order = np.argsort(key[:k], kind="mergesort")
            for o in range(k):
                c = cells[idx[order[o]]]
                if have[h] >= L:
                    done = True
                    break
                if cultivator[c] >= 0:
                    continue
                if gi == 0:
                    owner[c] = h
                elif gi == 2:
                    if reclaimed >= reclaim_max:
                        break
                    owner[c] = h
                    use[c] = crop[c]
                    reclaim_left[c] = reclaim_py[c] * area[c]
                    reclaimed += area[c]
                cultivator[c] = h
                have[h] += labor_per_ha[c] * area[c]
