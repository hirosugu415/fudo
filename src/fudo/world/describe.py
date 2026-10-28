"""「今回の世界」の説明（SPEC 第4.0節）。構造化データと日本語の文章を作る。シミュレーションには影響しない。"""

from __future__ import annotations

from collections import deque

import numpy as np

from .. import kernels
from ..grid import EDGE_LABELS, EDGES, Grid, region_label

KOPPEN_NAMES = {
    "Af": "熱帯雨林気候", "Am": "熱帯モンスーン気候", "Aw": "サバナ気候", "As": "サバナ気候（夏に乾燥）",
    "BW": "砂漠気候", "BS": "ステップ気候",
    "Cfa": "温暖湿潤気候", "Cfb": "西岸海洋性気候", "Cfc": "西岸海洋性気候（冷涼）",
    "Csa": "地中海性気候", "Csb": "地中海性気候（冷涼な夏）", "Csc": "地中海性気候（寒冷な夏）",
    "Cwa": "温暖冬季少雨気候", "Cwb": "温暖冬季少雨気候（冷涼な夏）", "Cwc": "温暖冬季少雨気候（寒冷な夏）",
    "D": "亜寒帯（冷帯）気候", "ET": "ツンドラ気候", "EF": "氷雪気候",
}
KIND_LABELS = {"sea": "海", "mountain": "山地", "plain": "平地"}
CONN_LABELS = {"river": "外から流れ込む川の上流", "sea": "海", "pass": "峠", "plain": "平地の道"}
MONTHS = [f"{m}月" for m in range(1, 13)]


def bearing_label(deg: float) -> str:
    names = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    return names[int(((deg % 360.0) + 22.5) // 45.0) % 8]


def koppen(T: np.ndarray, P: np.ndarray, kc) -> str:
    """月平均気温 T [℃] と月降水量 P [mm]（各12個）からケッペンの気候区分を判定する（北半球）。"""
    tmin, tmax, tann, pann = float(T.min()), float(T.max()), float(T.mean()), float(P.sum())
    summer = P[3:9]
    winter = np.concatenate([P[:3], P[9:]])
    if tmax < float(kc.polar_max_c):
        return "ET" if tmax >= float(kc.tundra_min_c) else "EF"
    dt = kc.dry_threshold
    share = float(dt.season_share)
    if summer.sum() >= share * pann:
        add = float(dt.summer_rain)
    elif winter.sum() >= share * pann:
        add = float(dt.winter_rain)
    else:
        add = float(dt.even)
    pth = float(dt.per_c) * tann + add
    if pann < pth:
        return "BW" if pann < pth / 2.0 else "BS"
    if tmin >= float(kc.tropical_min_c):
        pmin = float(P.min())
        if pmin >= float(kc.tropical_wet_mm):
            return "Af"
        if pann >= float(kc.monsoon_coef) * (float(kc.monsoon_offset_mm) - pmin):
            return "Am"
        return "As" if summer.min() < winter.min() else "Aw"
    if tmin <= float(kc.temperate_min_c):
        return "D"
    if summer.min() < winter.min() and winter.max() > float(kc.dry_summer_ratio) * summer.min() \
            and summer.min() < float(kc.dry_summer_max_mm):
        second = "s"
    elif winter.min() < summer.min() and summer.max() > float(kc.dry_winter_ratio) * winter.min():
        second = "w"
    else:
        second = "f"
    warm = int((T >= float(kc.warm_month_c)).sum())
    if tmax >= float(kc.hot_summer_c):
        third = "a"
    elif warm >= int(kc.warm_months_b):
        third = "b"
    else:
        third = "c"
    return "C" + second + third


def _lake_components(lake: np.ndarray) -> list[np.ndarray]:
    ny, nx = lake.shape
    seen = np.zeros_like(lake)
    comps = []
    for start in np.flatnonzero(lake.ravel()):
        if seen.flat[start]:
            continue
        q = deque([int(start)])
        seen.flat[start] = True
        cells = []
        while q:
            i = q.popleft()
            cells.append(i)
            r, c = divmod(i, nx)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ny and 0 <= cc < nx and lake[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append(rr * nx + cc)
        comps.append(np.array(cells, dtype=np.int64))
    return comps


def _rivers(grid: Grid, info: dict, arrays: dict, count: int, qmin: float) -> list[dict]:
    rcv = arrays["receiver"].astype(np.int64)
    sea = arrays["sea"].ravel()
    q = arrays["mean_discharge_m3s"].ravel()
    land = ~sea
    n = grid.n
    # 河口：流れ先がマップ外か海になる陸のセル
    mouth = land & ((rcv < 0) | np.where(rcv >= 0, sea[np.maximum(rcv, 0)], False))
    # 最も流量の多い上流側の隣（本流をたどるため）
    best = np.full(n, -1, dtype=np.int64)
    best_q = np.full(n, -1.0)
    for i in np.flatnonzero(land & (rcv >= 0)):
        r = rcv[i]
        if q[i] > best_q[r]:
            best_q[r] = q[i]
            best[r] = i
    # 外から流れ込む川がどの河口に行くか
    inflow_mouth = {}
    for k, inf in enumerate(info["inflows"]):
        i = inf["cell"]
        guard = 0
        while not mouth[i] and rcv[i] >= 0 and guard < n:
            i = rcv[i]
            guard += 1
        inflow_mouth.setdefault(int(i), []).append(k)
    out = []
    for m in np.flatnonzero(mouth)[np.argsort(-q[mouth], kind="stable")][:count]:
        if q[m] < qmin:
            break
        length = 0.0
        i = int(m)
        src = i
        while best[i] >= 0 and q[best[i]] >= qmin:
            j = int(best[i])
            dr, dc = divmod(j, grid.nx)
            r0, c0 = divmod(i, grid.nx)
            length += grid.dx * float(np.hypot(dr - r0, dc - c0))
            i = j
            src = i
        mr, mc = divmod(int(m), grid.nx)
        sr, sc = divmod(src, grid.nx)
        to_sea = rcv[m] >= 0 and sea[rcv[m]]
        if to_sea:
            dest = "海に注ぐ"
        else:
            e = ("north" if mr == 0 else "south" if mr == grid.ny - 1 else "west" if mc == 0 else "east")
            dest = f"{EDGE_LABELS[e]}の端からマップの外へ出る"
        inflow_edges = sorted({info["inflows"][k]["edge"] for k in inflow_mouth.get(int(m), [])}, key=EDGES.index)
        out.append({
            "mouth_row": mr, "mouth_col": mc, "mouth_region": region_label(mr, mc, grid),
            "source_row": sr, "source_col": sc, "source_region": region_label(sr, sc, grid),
            "mean_discharge_m3s": float(q[m]), "main_stem_km": length / 1000.0, "destination": dest,
            "from_outside": inflow_edges,
            "drainage_area_km2": float(arrays["drainage_area_m2"].ravel()[m]) / 1e6,
        })
    return out


def describe_world(gen, grid: Grid, info: dict, arrays: dict) -> dict:
    dcfg = gen.description
    h = arrays["elevation"]
    sea = arrays["sea"]
    lake = arrays["lake"]
    land = ~sea
    T = arrays["T_norm"]
    P = arrays["P_norm"]
    lines: list[str] = []

    # 気候
    low = land & (h <= np.quantile(h[land], float(dcfg.lowland_quantile))) if land.any() else land
    T_low = T[:, low].mean(axis=1)
    P_low = P[:, low].mean(axis=1)
    p_ann_land = P[:, land].sum(axis=0)
    code = koppen(T_low, P_low, dcfg.koppen)
    snow_months = [MONTHS[m] for m in range(12) if T_low[m] < float(gen.climate.precipitation.snow_threshold_c)]
    climate = {
        "latitude": info["latitude"],
        "koppen": code,
        "koppen_name": KOPPEN_NAMES.get(code, KOPPEN_NAMES.get(code[:1], code)),
        "lowland_annual_mean_c": float(T_low.mean()),
        "lowland_coldest_c": float(T_low.min()),
        "lowland_warmest_c": float(T_low.max()),
        "lowland_annual_precip_mm": float(P_low.sum()),
        "land_annual_precip_range_mm": [float(p_ann_land.min()), float(p_ann_land.max())],
        "lowland_monthly_precip_mm": [float(x) for x in P_low],
        "lowland_snow_months": snow_months,
    }
    w = info["wind"]
    lines.append(f"■ 気候：北緯 {info['latitude']:.1f} 度。{climate['koppen_name']}（{code}）。")
    lines.append(
        f"  低地の年平均気温 {climate['lowland_annual_mean_c']:.1f}℃"
        f"（最寒月 {climate['lowland_coldest_c']:.1f}℃、最暖月 {climate['lowland_warmest_c']:.1f}℃）。"
    )
    lines.append(
        f"  年降水量は低地で約 {climate['lowland_annual_precip_mm']:.0f} mm、"
        f"マップ内の陸で {p_ann_land.min():.0f}〜{p_ann_land.max():.0f} mm。"
    )
    wet = int(np.argmax(P_low))
    dry = int(np.argmin(P_low))
    lines.append(
        "  月降水量（低地）：" + " ".join(f"{m} {v:.0f}" for m, v in zip(MONTHS, P_low))
        + f"。最も多いのは{MONTHS[wet]}、少ないのは{MONTHS[dry]}。"
    )
    rs = info["rain_systems"]
    lines.append(f"  梅雨前線の雨は最盛期に月 +{rs['baiu']:.0f} mm 程度、台風の雨は月 +{rs['typhoon']:.0f} mm 程度（緯度で弱まる前の値）。")
    lines.append(
        f"  夏の季節風は{bearing_label(w['monsoon_summer_from_deg'])}から、冬は"
        f"{bearing_label(w['monsoon_summer_from_deg'] + 180.0)}から吹く。"
        + (f"低地で雪が降るのは {'・'.join(snow_months)}。" if snow_months else "低地ではほとんど雪が降らない。")
    )

    # 端と外側
    lines.append("■ マップの端：")
    for e in EDGES:
        ed = info["edges"][e]
        txt = f"  {EDGE_LABELS[e]}：外側は{KIND_LABELS[ed['kind']]}"
        if ed["kind"] != "sea":
            txt += f"（海まで約 {ed['outer_sea_km']:.0f} km）"
        lines.append(txt + "。")
    if info["gorge"] is not None:
        lines.append(f"  まわりを山に囲まれた盆地で、{EDGE_LABELS[info['gorge']['edge']]}の峡谷だけが水の出口になっている。")

    # 地形：山
    pk = dcfg.peaks
    radius = max(1, int(round(float(pk.isolation_radius_km) * 1000.0 / grid.dx)))
    valid = (land & (h >= float(pk.min_elevation_m))).ravel()
    peaks_mask = kernels.local_maxima(h.ravel(), grid.nx, grid.ny, radius, valid)
    idx = np.flatnonzero(peaks_mask)
    idx = idx[np.argsort(-h.ravel()[idx], kind="stable")][: int(pk.count)]
    peaks = []
    for i in idx:
        r, c = divmod(int(i), grid.nx)
        peaks.append({"row": r, "col": c, "elevation_m": float(h[r, c]), "region": region_label(r, c, grid)})
    land_share = float(land.mean())
    al_share = float((arrays["alluvium_m"][land] > float(gen.geology.alluvium.min_thickness_m)).mean()) if land.any() else 0.0
    lines.append(
        f"■ 地形：陸 {land_share * 100:.0f}%、最高地点 {h.max():.0f} m。"
        f"陸のうち沖積層（川が運んだ土砂の平地）が {al_share * 100:.0f}%。"
    )
    if peaks:
        lines.append("  主な山：" + "、".join(f"{p['region']}（{p['elevation_m']:.0f} m）" for p in peaks) + "。")

    # 川
    rivers = _rivers(grid, info, arrays, int(dcfg.rivers.count), float(gen.hydrology.river_min_discharge_m3s))
    if rivers:
        lines.append("■ 主な川（年平均流量の多い順）：")
        for rv in rivers:
            src = (f"{'・'.join(EDGE_LABELS[e] for e in rv['from_outside'])}の端の外から流れ込み、"
                   if rv["from_outside"] else f"{rv['source_region']}から流れ出し、")
            lines.append(
                f"  {src}{rv['mouth_region']}で{rv['destination']}。"
                f"年平均流量 {rv['mean_discharge_m3s']:.1f} m³/s、本流 {rv['main_stem_km']:.1f} km、"
                f"流域 {rv['drainage_area_km2']:.0f} km²（マップ内）。"
            )
    if info["inflows"]:
        lines.append(
            "  外から流れ込む川：" + "、".join(
                f"{EDGE_LABELS[i['edge']]}の端（外側の流域 {i['area_km2']:.0f} km²）" for i in info["inflows"]
            ) + "。"
        )

    # 湖
    lakes = []
    for comp in _lake_components(lake):
        a = comp.size * grid.cell_area / 1e6
        if a < float(dcfg.lakes.min_area_km2):
            continue
        rr, cc = np.divmod(comp, grid.nx)
        depth = float((arrays["filled"].ravel()[comp] - h.ravel()[comp]).max())
        lakes.append({"area_km2": a, "max_depth_m": depth,
                      "region": region_label(float(rr.mean()), float(cc.mean()), grid)})
    lakes.sort(key=lambda x: -x["area_km2"])
    lakes = lakes[: int(dcfg.lakes.count)]
    if lakes:
        lines.append("■ 湖：" + "、".join(f"{l['region']}（{l['area_km2']:.2f} km²、深さ {l['max_depth_m']:.0f} m）" for l in lakes) + "。")

    # 資源
    lines.append("■ 鉱物資源：")
    by_kind: dict[str, list[dict]] = {}
    for d in info["deposits"]:
        by_kind.setdefault(d["mineral"], []).append(d)
    resources = {}
    for dep_cfg in gen.minerals.deposits:
        ds = sorted(by_kind.get(dep_cfg.name, []), key=lambda d: -d["reserve_t"])
        total = float(sum(d["reserve_t"] for d in ds))
        resources[dep_cfg.name] = {"label": dep_cfg.label, "count": len(ds), "total_reserve_t": total}
        if not ds:
            lines.append(f"  {dep_cfg.label}：なし。")
            continue
        top = ds[: int(dcfg.resources.max_per_kind)]
        lines.append(
            f"  {dep_cfg.label}：{len(ds)} か所、計 {total / 1e4:,.0f} 万t。"
            + "、".join(
                f"{region_label(d['row'], d['col'], grid)}（{d['reserve_t'] / 1e4:,.0f} 万t、品位 {d['grade']:.3g}、"
                f"深さ {d['depth_top_m']:.0f} m〜）" for d in top
            ) + "。"
        )
    ft = arrays["forest_type"]
    forest = {}
    parts = []
    for k, t in enumerate(gen.forest.types):
        share = float((ft[land] == k).mean()) if land.any() else 0.0
        forest[t.name] = share
        if share > 0:
            parts.append(f"{t.label} {share * 100:.0f}%")
    lines.append("■ 森林（陸に占める割合）：" + ("、".join(parts) if parts else "なし") + "。")
    fk = arrays["fish_kind"]
    km2 = grid.cell_area / 1e6
    fish = {"coastal_km2": float((fk == 0).sum() * km2), "lake_km2": float((fk == 1).sum() * km2),
            "river_cells": int((fk == 2).sum())}
    lines.append(
        f"■ 漁場：沿岸 {fish['coastal_km2']:.1f} km²、湖 {fish['lake_km2']:.2f} km²、"
        f"川 約 {fish['river_cells'] * grid.dx / 1000:.0f} km。"
    )

    # 外部世界
    ex = info["external"]
    comp = ex["composition"]
    lines.append(
        f"■ 外部世界：地形は山地 {comp['mountain'] * 100:.0f}%・平地 {comp['plain'] * 100:.0f}%・"
        f"島 {comp['island'] * 100:.0f}%。人口成長の傾向 年 {ex['population_growth_per_year'] * 100:.1f}%、"
        f"景気変動の大きさ {ex['business_cycle_amplitude'] * 100:.1f}%。"
    )
    for c in ex["cities"]:
        lines.append(
            f"  外部の大都市（人口 約 {c['population'] / 1e4:,.0f} 万人）が、{EDGE_LABELS[c['edge']]}の端の"
            f"{CONN_LABELS[c['kind']]}（{region_label(c['row'], c['col'], grid)}）でつながっている。"
        )

    return {
        "text": "\n".join(lines),
        "climate": climate,
        "peaks": peaks,
        "rivers": rivers,
        "lakes": lakes,
        "resources": resources,
        "forest_share": forest,
        "fishery": fish,
    }
