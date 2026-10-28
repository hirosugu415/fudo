"""社会の1ステップ（ウォームアップは1年、本番は1か月）。

各主体は自分の利益・効用だけで判断する（SPEC 原則1・2）。ここにあるのは、
- 世帯：どこに住むか（ロジット）、何を仕事にするか、どの田畑を開墾・借りる・手放すか、どの店で買うか
- 店：開くか（見込み利益のロジット）、やめるか
- 市場：小作料・地代・米の値段（需給と輸送費から決まる）
- 自然の結果：収穫、漁獲、出生・死亡、道（通行量の結果）
だけで、町や道を上から置く処理はない。
"""

from __future__ import annotations

import numpy as np

from .. import kernels
from .. import rng as rngmod
from ..grid import Grid
from . import market
from .state import (
    OCC_FARM,
    OCC_FISH,
    OCC_NONE,
    OCC_SHOP,
    SHOP_DAILY,
    SHOP_INN,
    SHOP_KINDS,
    SHOP_SPECIALTY,
    USE_FIELD,
    USE_PADDY,
    USE_WILD,
    Households,
    Land,
)
from .transport import Transport, path_cells, traveler_stops, update_roads

CROPS = ((USE_PADDY, "paddy"), (USE_FIELD, "field"))


def box_sum(a2: np.ndarray, r: int) -> np.ndarray:
    """半径 r マスの正方形の範囲の合計（累積和で計算）。"""
    if r <= 0:
        return a2.copy()
    k = 2 * r + 1
    pad = np.pad(a2, r)
    cs = np.pad(np.cumsum(np.cumsum(pad, axis=0), axis=1), ((1, 0), (1, 0)))
    return cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]


class Society:
    def __init__(self, arrays: dict, info: dict, grid: Grid, beh, gen, run_seed: int):
        self.b = beh
        self.grid = grid
        self.nx, self.ny, n = grid.nx, grid.ny, grid.n
        self.n_cells = n
        self.area_ha = grid.cell_area / 1e4
        self.rows = np.arange(n) // grid.nx
        self.cols = np.arange(n) % grid.nx

        self.sea = arrays["sea"].ravel()
        self.lake = arrays["lake"].ravel()
        self.passable = ~self.sea & ~self.lake
        self.elev = arrays["elevation"].ravel()

        # ---- 田畑に向くか（自然から決まる、変化しない部分）----
        fcfg = beh.farming
        T = arrays["T_norm"]
        wi = np.maximum(T - float(gen.forest.base_c), 0.0).sum(axis=0).ravel()
        gy, gx = np.gradient(arrays["elevation"], grid.dx)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).ravel()
        fert = arrays["soil_fertility"].ravel().astype(np.float64)
        drain = arrays["soil_drainage"].ravel().astype(np.float64)
        irr = arrays["irrigation"].ravel().astype(np.float64)
        self.pot = np.zeros(n)            # kg/ha/年（平年・働き手が足りているとき）
        self.crop = np.zeros(n, dtype=np.int8)
        self.labor_per_ha = np.zeros(n)
        self.reclaim_py = np.zeros(n)     # 人年/ha
        self.irrigated = np.zeros(n, dtype=bool)
        for code, name in CROPS:
            c = fcfg.crops[name]
            ok = (self.passable & (slope_deg <= float(c.max_slope_deg)) & (irr >= float(c.min_irrigation))
                  & (drain <= float(c.max_drainage)))
            clim = np.clip((wi - float(c.wi_min)) / (float(c.wi_full) - float(c.wi_min)), 0.0, 1.0)
            water = irr if code == USE_PADDY else 1.0
            y = np.where(ok, float(c.base_yield_kg_per_ha) * fert * clim * water, 0.0)
            better = y > self.pot
            self.pot = np.where(better, y, self.pot)
            self.crop = np.where(better, code, self.crop).astype(np.int8)
            self.labor_per_ha = np.where(better, float(c.labor_per_ha), self.labor_per_ha)
            self.reclaim_py = np.where(better, float(c.reclaim_person_years_per_ha), self.reclaim_py)
        self.irrigated = self.crop == USE_PADDY
        self.arable = self.pot > 0.0

        # ---- 洪水の起きやすさ ----
        fl = beh.households.flood
        q = arrays["mean_discharge_m3s"].ravel()
        src = (q >= float(fl.river_min_discharge_m3s)) & self.passable
        d, near = kernels.nearest_source(src, grid.nx, grid.ny, grid.dx)
        head = np.where(near >= 0, self.elev - self.elev[np.maximum(near, 0)], np.inf)
        self.flood = np.where(
            near >= 0,
            np.exp(-np.maximum(head, 0.0) / float(fl.height_scale_m)) * np.exp(-d / (float(fl.distance_scale_km) * 1000.0)),
            0.0,
        )

        # ---- 漁場 ----
        self.fish_kind = arrays["fish_kind"].ravel()
        self.fish_cap = arrays["fish_capacity_t"].ravel().astype(np.float64)
        self.fish_r = arrays["fish_growth_per_yr"].ravel().astype(np.float64)

        # ---- 外部の大都市がつながる地点 ----
        raw = np.array([c["row"] * grid.nx + c["col"] for c in info["external"]["cities"]], dtype=np.int64)
        # 接続点が海や湖のマスなら、最も近い通れるマス（海の接続点なら港になる海辺）に置き直す
        _, nearest_land = kernels.nearest_source(self.passable, grid.nx, grid.ny, grid.dx)
        self.conn_cells = np.where(self.passable[raw], raw, nearest_land[raw]).astype(np.int64)
        self.conn_pop = np.array([c["population"] for c in info["external"]["cities"]], dtype=np.float64)

        # ---- 状態 ----
        self.hh = Households(int(beh.demography.max_members))
        self.land = Land(n, float(beh.land.residential_rent.initial_yen))
        self.land.fish_stock = self.fish_cap.copy()
        self.transport = Transport(arrays, grid.dx, beh.transport, beh.outside, gen.climate, gen.hydrology)
        self.tenancy_share = float(fcfg.tenancy.initial_rent_share)
        self.rice_export = True
        self.spend = {k: np.zeros(0) for k in ("daily", "specialty")}
        self.shop_profit = np.zeros(0)
        self.trees: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cand_sites = np.zeros(0, dtype=np.int64)
        self.stops = np.zeros(n)
        self.traveler_paths: list[tuple[np.ndarray, float]] = []
        self.flows: dict = {}
        self.sell_market = np.full(n, -1, dtype=np.int64)
        self.months_since_transport = 10**9
        self.entry_options = None
        self.stats_year = self._empty_stats()
        self.rng = {name: rngmod.run_stream(run_seed, f"society.{name}")
                    for name in ("demography", "formation", "migration", "review", "commerce")}

        o = beh.outside
        self.p_rice_ext = float(o.prices.rice_yen_per_kg)
        self.p_fish_ext = float(o.prices.fish_yen_per_kg)
        self.vot = self.transport.vot
        self.p_rice = np.full(n, self.p_rice_ext)
        self.p_fish = np.full(n, self.p_fish_ext)
        self.freight_to_conn = np.full(n, np.inf)
        self.access = np.zeros(n)

        st = beh.households.settlement
        self.cap_hh = min(int(st.max_households_per_cell), int(self.area_ha / float(st.residential_area_ha)))
        self.r_cult = max(1, int(round(float(fcfg.cultivation_radius_km) * 1000.0 / grid.dx)))
        self.r_fish = max(1, int(round(float(beh.fishing.fishing_radius_km) * 1000.0 / grid.dx)))
        self.r_social = max(1, int(round(float(st.social_radius_km) * 1000.0 / grid.dx)))
        B = int(beh.transport.roads.market_block_cells)
        nbx = (grid.nx + B - 1) // B
        self.block = (self.rows // B) * nbx + (self.cols // B)

    # ------------------------------------------------------------------
    # 小さな道具
    # ------------------------------------------------------------------
    def _empty_stats(self) -> dict:
        return {"births": 0, "deaths": 0, "immigrants": 0, "emigrants": 0, "new_households": 0,
                "shops_opened": 0, "shops_closed": 0}

    def _labor(self, ids) -> np.ndarray:
        f = self.b.farming
        a = self.hh.ages[ids]
        lo, hi = int(f.adult_age[0]), int(f.adult_age[1])
        adult = ((a >= lo) & (a <= hi)).sum(axis=1)
        part = (((a >= int(f.child_work_age)) & (a < lo)) | ((a > hi) & (a <= int(f.elder_work_max_age)))).sum(axis=1)
        return adult + float(f.labor_per_child_or_elder) * part

    def _res_count(self) -> np.ndarray:
        ids = self.hh.ids()
        return np.bincount(self.hh.cell[ids], minlength=self.n_cells)

    def _farm_area(self, res_count) -> np.ndarray:
        st = self.b.households.settlement
        shops = self._shop_counts().sum(axis=0)
        used = res_count * float(st.residential_area_ha) + shops * float(self.b.commerce.land_per_shop_ha)
        return np.clip(self.area_ha - used, 0.0, None)

    def _shop_counts(self) -> np.ndarray:
        n = self.hh.n
        sel = self.hh.alive[:n] & (self.hh.occ[:n] == OCC_SHOP)
        idx = self.hh.shop_kind[:n][sel].astype(np.int64) * self.n_cells + self.hh.cell[:n][sel]
        return np.bincount(idx, minlength=len(SHOP_KINDS) * self.n_cells).reshape(len(SHOP_KINDS), self.n_cells)

    def _market_shops(self, shops) -> np.ndarray:
        """近くの店を1つの市場にまとめる：market_block_cells 四方の中の店を、その中で最も店の多いマス（同数なら番号の小さいマス）に集める。
        買い物や経路の計算は市場ごとに行う（店のあるマスごとに経路を計算すると重すぎるため）。"""
        tot = shops.sum(axis=0)
        cells = np.flatnonzero(tot > 0)
        out = np.zeros_like(shops)
        if cells.size == 0:
            return out
        order = np.lexsort((cells, -tot[cells], self.block[cells]))
        sc = cells[order]
        blk = self.block[sc]
        first = np.concatenate([[True], blk[1:] != blk[:-1]])
        rep_of_block = dict(zip(blk[first].tolist(), sc[first].tolist()))
        rep_cell = np.array([rep_of_block[b] for b in self.block[cells]], dtype=np.int64)
        for k in range(shops.shape[0]):
            np.add.at(out[k], rep_cell, shops[k, cells])
        return out

    def _dist_km(self, a, b) -> np.ndarray:
        dr = self.rows[a] - self.rows[b]
        dc = self.cols[a] - self.cols[b]
        return np.hypot(dr, dc) * self.grid.dx / 1000.0

    def _window(self, center: int, r: int) -> np.ndarray:
        r0, c0 = divmod(int(center), self.nx)
        rr = np.arange(max(r0 - r, 0), min(r0 + r + 1, self.ny))
        cc = np.arange(max(c0 - r, 0), min(c0 + r + 1, self.nx))
        return (rr[:, None] * self.nx + cc[None, :]).ravel()

    def _release(self, h: int, keep_center: int | None) -> None:
        """世帯 h が耕している田畑のうち、keep_center から耕作圏の外にあるものを手放す（貸しに出す）。"""
        plots = np.flatnonzero(self.land.cultivator == h)
        if plots.size == 0:
            return
        if keep_center is not None:
            far = self._dist_km(plots, np.full(plots.size, keep_center)) > float(self.b.farming.cultivation_radius_km)
            plots = plots[far]
        self.land.cultivator[plots] = -1

    def _remove_household(self, h: int) -> None:
        """世帯がいなくなった（全員死亡・転出）。所有していた土地は持ち主がいなくなる。"""
        self.hh.alive[h] = False
        owned = np.flatnonzero(self.land.owner == h)
        # 借りて耕している人がいれば、その人のものになる（耕作者に帰属）。いなければ持ち主のいない土地
        tenants = self.land.cultivator[owned]
        self.land.owner[owned] = np.where((tenants >= 0) & (tenants != h), tenants, -1)
        self.land.cultivator[self.land.cultivator == h] = -1

    # ------------------------------------------------------------------
    # 交通：経路・価格・通行量・道
    # ------------------------------------------------------------------
    def _transport_update(self, years: float) -> None:
        tcfg = self.b.transport
        if self.trees:
            traffic = self._traffic()
            self.land.road = update_roads(self.land.road, traffic, tcfg.roads, years)
        T = self.transport
        T.update_costs(self.land.road)
        self.trees = {}
        self.conn_freight = []
        for c in self.conn_cells:
            self.trees[int(c)] = T.person_tree(int(c), np.inf)
            self.conn_freight.append(T.freight_tree(int(c)))
        if self.conn_freight:
            fd = np.stack([d for d, _ in self.conn_freight])
            self.freight_to_conn = fd.min(axis=0)
            self.best_conn = fd.argmin(axis=0)
        cut = float(tcfg.roads.shop_cutoff_hours)
        shops = self._shop_counts()
        mshops = self._market_shops(shops)
        for c in np.flatnonzero(mshops.sum(axis=0) > 0):
            if int(c) not in self.trees:
                self.trees[int(c)] = T.person_tree(int(c), cut)
        self.cand_sites = self._candidate_sites(shops)
        for c in self.cand_sites:
            if int(c) not in self.trees:
                self.trees[int(c)] = T.person_tree(int(c), cut)
        self._travelers()
        self._prices(mshops)
        self._access(mshops)
        self.months_since_transport = 0
        self.entry_options = None   # 候補地と経路が変わったので、出店の見込みは計算し直す

    def _candidate_sites(self, shops) -> np.ndarray:
        """新しく店を開く場所の候補：近くの世帯が多く、人通りの多いマス（店の人が自分で見込みを比べるための候補）。"""
        cc = self.b.commerce.entry
        r = max(1, int(round(float(cc.site_score_radius_km) * 1000.0 / self.grid.dx)))
        res = self._res_count().reshape(self.ny, self.nx).astype(np.float64)
        near = box_sum(res, r).ravel()
        score = np.log1p(near) + np.log1p(self.last_traffic if hasattr(self, "last_traffic") else 0.0)
        has_market = np.zeros(self.block.max() + 1, dtype=bool)
        has_market[self.block[shops.sum(axis=0) > 0]] = True
        score = np.where(self.passable & ~has_market[self.block] & (near > 0), score, -np.inf)
        order = np.argsort(-score, kind="stable")
        picked = []
        sp = int(cc.min_site_spacing_cells)
        for c in order:
            if not np.isfinite(score[c]) or len(picked) >= int(cc.candidate_sites):
                break
            if all(max(abs(self.rows[c] - self.rows[p]), abs(self.cols[c] - self.cols[p])) >= sp for p in picked):
                picked.append(int(c))
        return np.array(picked, dtype=np.int64)

    def _prices(self, shops) -> None:
        """米・魚の値段：外部の値段と、外部とつながる地点までの運賃・売りに行く手間から決まる（SPEC 10 の財市場）。"""
        sell_kg = float(self.b.transport.roads.sell_trip_kg)
        n = self.n_cells
        sites = [int(c) for c in np.flatnonzero(shops.sum(axis=0) > 0)]
        fc = self.freight_to_conn
        sign = -1.0 if self.rice_export else 1.0
        # 外部とつながる地点まで自分で運ぶ場合
        best = self.p_rice_ext + sign * fc
        self.sell_market = np.full(n, -1, dtype=np.int64)
        for c in sites:
            dist = self.trees[c][0]
            trip = 2.0 * dist * self.vot / sell_kg
            p = self.p_rice_ext + sign * fc[c] - trip if self.rice_export else self.p_rice_ext + fc[c] + trip
            better = (p > best) if self.rice_export else (p < best)
            better &= np.isfinite(dist)
            best = np.where(better, p, best)
            self.sell_market[better] = c
        self.p_rice = np.where(np.isfinite(best), np.maximum(best, 0.0), 0.0)
        pf = self.p_fish_ext - fc
        self.p_fish = np.where(np.isfinite(pf), np.maximum(pf, 0.0), 0.0)

    def _access(self, shops) -> None:
        tau = float(self.b.households.utility.access_time_scale_hours)
        acc = np.zeros(self.n_cells)
        for c in np.flatnonzero(shops[:SHOP_INN].sum(axis=0) > 0):
            d = self.trees[int(c)][0]
            acc += shops[:SHOP_INN, c].sum() * np.where(np.isfinite(d), np.exp(-d / tau), 0.0)
        per = float(self.b.commerce.outside_shops.shops_per_10k_people)
        for k, c in enumerate(self.conn_cells):
            d = self.trees[int(c)][0]
            acc += per * self.conn_pop[k] / 1e4 * np.where(np.isfinite(d), np.exp(-d / tau), 0.0)
        self.access = acc

    def _travelers(self) -> None:
        """外部の大都市どうしを行き来する旅人。経路の上で休む・泊まるマスを数える（宿・茶屋の需要）。"""
        icfg = self.b.commerce.inn
        self.stops = np.zeros(self.n_cells)
        self.traveler_paths = []
        T = self.transport
        for a in range(self.conn_cells.size):
            for b in range(a + 1, self.conn_cells.size):
                if self.conn_cells[a] == self.conn_cells[b]:
                    continue
                pred = self.trees[int(self.conn_cells[a])][1]
                path = path_cells(pred, int(self.conn_cells[b]))
                if path.size < 2 or path[-1] != self.conn_cells[a]:
                    continue
                travelers = float(icfg.travelers_per_year_per_100k_squared) * (self.conn_pop[a] / 1e5) * (self.conn_pop[b] / 1e5)
                for s in traveler_stops(path[::-1], T.person_hours, self.elev, T.crossing_hours, icfg, self.grid.dx):
                    self.stops[s] += travelers
                self.traveler_paths.append((path, travelers))

    def _traffic(self) -> np.ndarray:
        """前回の経路と、世帯・店の動き（買い物、米を売りに行く、品物の仕入れ、旅人）から、マスごとの年間の通行量を数える。"""
        tr = np.zeros(self.n_cells)
        for (cells, flows_by_site) in self.flows.values():
            for site, flow in flows_by_site:
                if site in self.trees:
                    kernels.add_path_traffic(self.trees[site][1], cells, flow, tr)
        # 米を売りに行く
        ids = self.hh.ids()
        if ids.size and hasattr(self, "surplus_kg"):
            sell_kg = float(self.b.transport.roads.sell_trip_kg)
            surplus = np.zeros(self.hh.n)
            k = min(self.hh.n, self.surplus_kg.size)
            surplus[:k] = self.surplus_kg[:k]   # 去年の収穫の後に増えた世帯は、まだ売る米がない
            per_cell = np.bincount(self.hh.cell[ids], weights=surplus[ids] / sell_kg * 2.0, minlength=self.n_cells)
            for c in np.unique(self.sell_market[per_cell > 0]):
                origins = np.flatnonzero((per_cell > 0) & (self.sell_market == c))
                if c >= 0 and int(c) in self.trees:
                    kernels.add_path_traffic(self.trees[int(c)][1], origins, per_cell[origins], tr)
                elif self.conn_cells.size:
                    for k, cc in enumerate(self.conn_cells):
                        sel = origins[self.best_conn[origins] == k]
                        kernels.add_path_traffic(self.trees[int(cc)][1], sel, per_cell[sel], tr)
        # 旅人
        for path, travelers in self.traveler_paths:
            tr[path] += travelers
        self.last_traffic = tr
        return tr

    # ------------------------------------------------------------------
    # 収穫・漁獲・収支
    # ------------------------------------------------------------------
    def _harvest_and_income(self, dT: float, pf: float) -> None:
        f = self.b.farming
        hh = self.hh
        ids = hh.ids()
        n_h = hh.n
        res = self._res_count()
        area = self._farm_area(res)
        # 冷害・干ばつ
        cold = max(0.0, -dT) * float(f.yield_temperature_sensitivity_per_c)
        dry = max(0.0, 1.0 - pf) * float(f.precip_shortfall_sensitivity)
        yfac = np.where(self.irrigated, 1.0 - cold, 1.0 - cold - dry)
        yfac = np.clip(yfac, 0.0, None)

        L = np.zeros(n_h)
        L[ids] = self._labor(ids)
        plots = np.flatnonzero((self.land.cultivator >= 0) & (self.land.use > 0))
        cult = self.land.cultivator[plots].astype(np.int64)
        alive = hh.alive[cult]
        plots, cult = plots[alive], cult[alive]
        farmed = self.land.reclaim_left[plots] <= 0.0
        a = area[plots]
        req = np.where(farmed, self.labor_per_ha[plots] * a, 0.0)
        dkm = self._dist_km(plots, hh.cell[cult])
        mult = 1.0 / (1.0 + float(f.distance_labor_loss_per_km) * dkm)
        R = np.bincount(cult, weights=req, minlength=n_h)
        Rm = np.bincount(cult, weights=req * mult, minlength=n_h)
        m_h = np.where(R > 0, Rm / np.maximum(R, 1e-12), 1.0)
        L_eff = L * m_h
        frac = np.where(R > 0, np.minimum(1.0, L_eff / np.maximum(R, 1e-12)), 0.0)
        y = np.where(farmed, self.pot[plots] * a * yfac[plots] * frac[cult] ** float(f.labor_exponent), 0.0)
        # 余った働き手で開墾を進める
        spare = np.maximum(L_eff - R, 0.0)
        reclaiming = ~farmed
        n_rec = np.bincount(cult[reclaiming], minlength=n_h)
        give = np.where(reclaiming, spare[cult] / np.maximum(n_rec[cult], 1), 0.0)
        self.land.reclaim_left[plots] = np.maximum(self.land.reclaim_left[plots] - give, 0.0)

        # 小作料
        owner = self.land.owner[plots].astype(np.int64)
        tenant = (owner >= 0) & (owner != cult)
        tenant &= hh.alive[np.maximum(owner, 0)]
        rent_kg = np.where(tenant, self.tenancy_share * y, 0.0)
        prod = np.bincount(cult, weights=y - rent_kg, minlength=n_h)
        rent_recv_kg = np.bincount(owner[tenant], weights=rent_kg[tenant], minlength=n_h)

        # 漁獲
        fish_kg = self._fishing(L)

        # 店の利益（前の年の支出から）
        shop_profit = self._shop_profits()

        # 収支
        p = self.p_rice[hh.cell[:n_h]]
        need = float(f.food_need_kg_per_person_year) * hh.members()
        own_food = np.minimum(prod, need)
        sold = np.maximum(prod - need, 0.0)
        money_in = sold * p + rent_recv_kg * p + fish_kg * self.p_fish[hh.cell[:n_h]] + shop_profit
        buy_kg = need - own_food
        cost = buy_kg * p
        rent_paid = self.land.res_rent[hh.cell[:n_h]]
        cash = hh.money[:n_h] + money_in - cost - rent_paid
        short = cash < 0.0
        afford_kg = np.where(short & (p > 0), np.maximum(hh.money[:n_h] + money_in - rent_paid, 0.0) / np.maximum(p, 1e-12), buy_kg)
        afford_kg = np.minimum(afford_kg, buy_kg)
        food = own_food + afford_kg
        hh.food_ratio[:n_h] = np.where(need > 0, food / np.maximum(need, 1e-12), 1.0)
        cash = np.maximum(cash, 0.0)
        disposable = np.maximum(money_in - cost - rent_paid, 0.0)
        g = self.b.commerce.goods
        self.spend = {k: np.zeros(n_h) for k in ("daily", "specialty")}
        for k in ("daily", "specialty"):
            s = float(g[k].budget_share) * disposable
            s = np.minimum(s, cash)
            self.spend[k] = s
            cash = cash - s
        hh.money[:n_h] = np.where(hh.alive[:n_h], cash, 0.0)
        hh.income[:n_h] = own_food * np.where(p > 0, p, self.p_rice_ext) + money_in - rent_paid
        self.surplus_kg = np.where(hh.alive[:n_h], sold, 0.0)
        self.production_kg = float(prod[hh.alive[:n_h]].sum() + rent_recv_kg.sum())
        self.need_kg = float(need[hh.alive[:n_h]].sum())
        self.rice_export = self.production_kg >= self.need_kg
        # 住む場所の地代は、そのマスの持ち主に入る
        owner_cell = self.land.owner[hh.cell[ids]]
        ok = (owner_cell >= 0) & hh.alive[np.maximum(owner_cell, 0)]
        np.add.at(hh.money, owner_cell[ok], rent_paid[ids][ok])

    def _fishing(self, L: np.ndarray) -> np.ndarray:
        """漁をする世帯の漁獲 [kg/年]。資源量はロジスティックに回復する。"""
        hh = self.hh
        n_h = hh.n
        out = np.zeros(n_h)
        ids = hh.ids()
        fishers = ids[hh.occ[ids] == OCC_FISH]
        stock = self.land.fish_stock
        cap = self.fish_cap
        catch_cell = np.zeros(self.n_cells)
        per_adult = float(self.b.fishing.catch_per_adult_at_capacity_kg)
        for h in fishers:
            win = self._window(hh.cell[h], self.r_fish)
            win = win[cap[win] > 0]
            if win.size == 0:
                continue
            dens = stock[win].sum() / cap[win].sum()
            kg = per_adult * L[h] * dens
            out[h] = kg
            catch_cell[win] += kg / 1000.0 * cap[win] / cap[win].sum()
        grow = self.fish_r * stock * (1.0 - stock / np.maximum(cap, 1e-12))
        self.land.fish_stock = np.clip(stock + grow - catch_cell, 0.0, cap)
        return out

    # ------------------------------------------------------------------
    # 商い
    # ------------------------------------------------------------------
    def _huff_inputs(self, kind: str):
        hh = self.hh
        ids = hh.ids()
        cells = hh.cell[ids]
        uc, inv = np.unique(cells, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        sp = np.zeros(hh.n)
        k = min(hh.n, self.spend[kind].size)
        sp[:k] = self.spend[kind][:k]   # 去年の収支の後に増えた世帯は、まだ支出がない
        tot = np.bincount(inv, weights=sp[ids]) if ids.size else np.zeros(0)
        per = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        return uc.astype(np.int64), cnt, per

    def _huff_setup(self, kind: str, mshops, extra_sites=()):
        """種類 kind の買い物の計算の準備：市場（と、出店を考える場所）ごとの店の数・運賃・世帯のいるマスまでの時間の表。
        外部の大都市の店も選択肢に入れる。"""
        g = self.b.commerce.goods[kind]
        k = SHOP_KINDS.index(kind)
        uc, cnt, per = self._huff_inputs(kind)
        sites = [int(c) for c in np.flatnonzero(mshops[k] > 0)]
        for c in extra_sites:
            if int(c) not in sites:
                sites.append(int(c))
        n_shop = [float(mshops[k, c]) for c in sites]
        frac = [float(g.weight_kg_per_yen) * self.freight_to_conn[c] for c in sites]
        per10k = float(self.b.commerce.outside_shops.shops_per_10k_people)
        all_sites = sites + [int(c) for c in self.conn_cells]
        n_shop += [per10k * p / 1e4 for p in self.conn_pop]
        frac += [0.0] * self.conn_cells.size
        times = np.full((len(all_sites), uc.size), np.inf)
        for i, c in enumerate(all_sites):
            if c in self.trees:
                times[i] = self.trees[c][0][uc]
        return {"g": g, "sites": all_sites, "n": np.array(n_shop), "frac": np.nan_to_num(np.array(frac), posinf=1e9),
                "times": times, "uc": uc, "cnt": cnt, "per": per}

    def _huff_eval(self, setup, extra_site: int = -1):
        g = setup["g"]
        site_idx = setup["sites"].index(extra_site) if extra_site >= 0 else -1
        return market.huff(setup["times"], setup["n"], setup["frac"], setup["cnt"], setup["per"],
                           float(g.trips_per_year), self.vot, float(g.logit_share), float(g.agglomeration), site_idx, 1.0)

    def _huff(self, kind: str, mshops, extra_site: int = -1):
        """種類 kind の店について、市場ごとの売上と世帯の往復の回数を計算する。"""
        setup = self._huff_setup(kind, mshops, (extra_site,) if extra_site >= 0 else ())
        rev, flows = self._huff_eval(setup, extra_site)
        return setup["sites"], rev, flows, setup["uc"]

    def _shop_profits(self) -> np.ndarray:
        """店の利益 [円/年]：前の年の世帯の支出から決まる売上 × 利幅 − 固定費。宿は旅人が休む回数から。"""
        hh = self.hh
        out = np.zeros(hh.n)
        ids = hh.ids()
        shop_ids = ids[hh.occ[ids] == OCC_SHOP]
        # 世帯の支出が変わったので、出店の見込みの一覧は次に計算し直す（店が1軒もなくても）
        self.entry_options = None
        if shop_ids.size == 0:
            self.flows = {}
            self.shop_profit = out
            return out
        shops = self._shop_counts()
        mshops = self._market_shops(shops)
        self.flows = {}
        shop_block = self.block[hh.cell[shop_ids]]
        for kind in ("daily", "specialty"):
            g = self.b.commerce.goods[kind]
            k = SHOP_KINDS.index(kind)
            sites, rev, flows, uc = self._huff(kind, mshops)
            self.flows[kind] = (uc, [(s, flows[i]) for i, s in enumerate(sites)])
            for i, s in enumerate(sites[: len(sites) - self.conn_cells.size]):
                members = shop_ids[(shop_block == self.block[s]) & (hh.shop_kind[shop_ids] == k)]
                if members.size:
                    out[members] = float(g.margin) * rev[i] / members.size - float(g.fixed_cost_yen_per_year)
        inn = self.b.commerce.inn
        inn_ids = shop_ids[hh.shop_kind[shop_ids] == SHOP_INN]
        if inn_ids.size:
            rev = self._inn_revenue(shops)
            for c in np.unique(hh.cell[inn_ids]):
                members = inn_ids[hh.cell[inn_ids] == c]
                out[members] = float(inn.margin) * rev[c] / members.size - float(inn.fixed_cost_yen_per_year)
        self.shop_profit = out
        self.entry_options = None   # 売上が変わったので、出店の見込みは次に計算し直す
        return out

    def _inn_revenue(self, shops, extra_site: int = -1) -> np.ndarray:
        """宿・茶屋の売上（マスごと）：旅人が休むマスから catchment_cells 以内の宿で、休む回数を等分する。"""
        icfg = self.b.commerce.inn
        r = int(icfg.catchment_cells)
        inns = shops[SHOP_INN].astype(np.float64).copy()
        if extra_site >= 0:
            inns[extra_site] += 1.0
        inn2 = inns.reshape(self.ny, self.nx)
        near = box_sum(inn2, r).ravel()
        demand = np.where(near > 0, self.stops / np.maximum(near, 1e-12), 0.0)
        per_inn = box_sum(demand.reshape(self.ny, self.nx), r).ravel()
        return per_inn * inns * float(icfg.spend_per_stop_yen)

    def _alternative_income(self, ids) -> np.ndarray:
        """店をやめて、今の場所で農業か漁業をした場合の見込みの所得。"""
        maps = self._opportunity_maps()
        L = self._labor(ids)
        y_farm, y_fish = self._expected_income(ids, self.hh.cell[ids], L, maps)
        return np.maximum(y_farm, y_fish)

    def _commerce_exit(self) -> None:
        hh = self.hh
        ids = hh.ids()
        shop_ids = ids[hh.occ[ids] == OCC_SHOP]
        if shop_ids.size == 0:
            return
        alt = self._alternative_income(shop_ids)
        prof = np.zeros(hh.n)
        k = min(hh.n, self.shop_profit.size)
        prof[:k] = self.shop_profit[:k]
        prof = prof[shop_ids]
        losing = prof < alt
        hh.loss_years[shop_ids] = np.where(losing, hh.loss_years[shop_ids] + 1, 0)
        quit_ = hh.loss_years[shop_ids] >= int(self.b.commerce.exit.loss_years)
        for h in shop_ids[quit_]:
            hh.occ[h] = OCC_NONE
            hh.shop_kind[h] = -1
            hh.loss_years[h] = 0
            self.stats_year["shops_closed"] += 1

    def _commerce_entry(self, dt: float) -> None:
        """商売を始めるか：今の所得と、店を開いた場合の見込みの利益を比べてロジットで選ぶ（SPEC 9.2 の参入判断）。"""
        cc = self.b.commerce
        hh = self.hh
        r = self.rng["commerce"]
        ids = hh.ids()
        if ids.size == 0 or not self.trees:
            return
        draws = r.random(ids.size)
        p = 1.0 - np.exp(-float(cc.entry.consider_prob_per_year) * dt)
        L = self._labor(ids)
        cand = ids[(draws < p) & (hh.occ[ids] != OCC_SHOP) & (hh.money[ids] >= float(cc.entry.start_money_yen)) & (L >= 1.0)]
        if cand.size == 0:
            return
        res = self._res_count()
        shops = self._shop_counts()
        if self.entry_options is None:
            self.entry_options = self._entry_options(shops, res)
        options = self.entry_options
        for h in cand:
            if not options:
                break
            vals = np.array([o[2] for o in options])
            base = hh.income[h]
            u = np.concatenate([[base], vals]) / float(cc.entry.logit_scale_yen)
            u = u - u.max()
            pr = np.exp(u)
            pr /= pr.sum()
            pick = int(r.choice(pr.size, p=pr))
            if pick == 0:
                continue
            kind, site, _ = options[pick - 1]
            if res[site] >= self.cap_hh:
                continue
            res[hh.cell[h]] -= 1
            res[site] += 1
            hh.cell[h] = site
            hh.occ[h] = OCC_SHOP
            hh.shop_kind[h] = kind
            hh.loss_years[h] = 0
            self._release(h, None)
            shops[kind, site] += 1
            self.stats_year["shops_opened"] += 1
            # この場所・種類の見込みを更新する（同じ場所に次々と出店しすぎないように）
            if kind == SHOP_INN:
                v = self._inn_entry_value(site, shops)
            else:
                setup = self.entry_setups[kind]
                setup["n"][setup["sites"].index(site)] += 1.0
                v = float(self._goods_entry_values(kind, setup, [site])[0])
            options = [o for o in options if not (o[0] == kind and o[1] == site)]
            options.append((kind, site, v))
        self.entry_options = options

    def _goods_entry_values(self, kind: int, setup, sites) -> np.ndarray:
        """種類 kind の店を、場所 sites のそれぞれに1軒増やしたときの、その店の見込みの利益 [円/年]（差分計算）。"""
        g = setup["g"]
        idx = np.array([setup["sites"].index(s) for s in sites], dtype=np.int64)
        ok = np.array([s in self.trees for s in sites])
        rev = market.huff_marginal(setup["times"], setup["n"], setup["frac"], setup["cnt"], setup["per"],
                                   float(g.trips_per_year), self.vot, float(g.logit_share), float(g.agglomeration), idx)
        val = float(g.margin) * rev / (setup["n"][idx] + 1.0) - float(g.fixed_cost_yen_per_year)
        return np.where(ok, val, -np.inf)

    def _inn_entry_value(self, site: int, shops) -> float:
        cc = self.b.commerce
        if site not in self.trees:
            return -np.inf
        rev = self._inn_revenue(shops, extra_site=site)
        return float(cc.inn.margin) * rev[site] / (shops[SHOP_INN, site] + 1) - float(cc.inn.fixed_cost_yen_per_year)

    def _entry_options(self, shops, res):
        """出店の選択肢（種類, 場所, 見込みの利益）。場所は既存の市場と候補地。買い物の計算の準備は種類ごとに1回だけ。"""
        mshops = self._market_shops(shops)
        sites = sorted(set(int(c) for c in np.flatnonzero(mshops.sum(axis=0) > 0)) | set(int(c) for c in self.cand_sites))
        sites = [s for s in sites if res[s] < self.cap_hh]
        self.entry_setups = {k: self._huff_setup(SHOP_KINDS[k], mshops, sites) for k in (SHOP_DAILY, SHOP_SPECIALTY)}
        opts = []
        vals = {k: self._goods_entry_values(k, self.entry_setups[k], sites) for k in (SHOP_DAILY, SHOP_SPECIALTY)}
        for i, site in enumerate(sites):
            for kind in range(len(SHOP_KINDS)):
                v = vals[kind][i] if kind != SHOP_INN else self._inn_entry_value(site, shops)
                if np.isfinite(v):
                    opts.append((kind, site, float(v)))
        return opts

    # ------------------------------------------------------------------
    # 住む場所・仕事の選択
    # ------------------------------------------------------------------
    def _opportunity_maps(self) -> dict:
        """マスごとの「そこに住んだら手に入りそうなもの」（どの世帯も同じ情報で判断する）。"""
        f = self.b.farming
        res = self._res_count()
        area = self._farm_area(res)
        land = self.land
        free_farm = (land.use > 0) & (land.owner < 0) & (land.cultivator < 0)
        wild = (land.use == USE_WILD) & self.arable & (land.owner < 0)
        lease = (land.use > 0) & (land.owner >= 0) & (land.cultivator < 0)
        disc = 1.0 - self.reclaim_py / (np.maximum(self.labor_per_ha, 1e-12) * float(f.reclaim_horizon_years))
        val = (self.pot * area * (free_farm + wild * np.clip(disc, 0.0, 1.0) + lease * (1.0 - self.tenancy_share)))
        avail = area * (free_farm | wild | lease)
        r = self.r_cult
        sh = (self.ny, self.nx)
        A = box_sum(avail.reshape(sh), r).ravel()
        V = box_sum(val.reshape(sh), r).ravel()
        lab = box_sum((self.labor_per_ha * avail).reshape(sh), r).ravel()
        yield_per_ha = np.where(A > 0, V / np.maximum(A, 1e-12), 0.0)
        labor_per_ha = np.where(A > 0, lab / np.maximum(A, 1e-12), 2.0)
        cap = box_sum(self.fish_cap.reshape(sh), self.r_fish).ravel()
        stock = box_sum(self.land.fish_stock.reshape(sh), self.r_fish).ravel()
        nfish = box_sum((self.fish_cap > 0).astype(np.float64).reshape(sh), self.r_fish).ravel()
        fish_ok = nfish >= int(self.b.fishing.min_fishery_cells)
        fish_per_adult = np.where(fish_ok & (cap > 0), float(self.b.fishing.catch_per_adult_at_capacity_kg) * stock / np.maximum(cap, 1e-12), 0.0)
        social = box_sum(res.reshape(sh).astype(np.float64), self.r_social).ravel()
        maps = {"A": A, "yield": yield_per_ha, "labor": labor_per_ha, "fish": fish_per_adult,
                "social": social, "res": res, "area": area}
        maps["far_cdf"] = self._far_cdf(maps)
        return maps

    def _expected_income(self, ids, cells, L, maps):
        """世帯 ids がマス cells に住んだ場合の、農業・漁業の見込みの所得 [円/年]。"""
        f = self.b.farming
        need = float(f.food_need_kg_per_person_year) * self.hh.members(ids)
        land_ha = np.minimum(L / np.maximum(maps["labor"][cells], 1e-12), maps["A"][cells])
        kg = land_ha * maps["yield"][cells]
        p = self.p_rice[cells]
        y_farm = np.minimum(kg, need) * self.p_rice_ext + np.maximum(kg - need, 0.0) * p
        fish = maps["fish"][cells] * L
        y_fish = fish * self.p_fish[cells] + 0.0
        rent = self.land.res_rent[cells]
        return y_farm - rent, y_fish - rent

    def _utility(self, ids, cells, L, maps, current, income_now=None, move_cost=True):
        u = self.b.households.utility
        y_farm, y_fish = self._expected_income(ids, cells, L, maps)
        y = np.maximum(y_farm, y_fish)
        occ = np.where(y_fish > y_farm, OCC_FISH, OCC_FARM)
        if income_now is not None:
            stay = cells == current
            y = np.where(stay, np.maximum(income_now, y), y)
        yref = float(u.income_ref_yen)
        floor = float(u.income_floor_yen)
        amen = (np.maximum(y, floor) / yref) ** float(u.amenity_income_elasticity)
        crowd = maps["res"][cells] / self.cap_hh
        val = (float(u.income) * np.log(np.maximum(y, floor) / yref)
               + amen * (float(u.access) * np.log1p(self.access[cells]) - float(u.flood) * self.flood[cells]
                         + float(u.social) * np.log1p(maps["social"][cells]))
               - float(u.crowding) * crowd
               - (float(u.moving_cost) * (cells != current) if move_cost else 0.0))
        val = np.where((maps["res"][cells] >= self.cap_hh) & (cells != current), -np.inf, val)
        val = np.where(self.passable[cells], val, -np.inf)
        return val, occ

    def _outside_utility(self) -> float:
        u = self.b.households.utility
        y = float(self.b.outside.outside_household.income_yen_per_year)
        return float(u.income) * np.log(max(y, float(u.income_floor_yen)) / float(u.income_ref_yen))

    def _sample_candidates(self, r, center: int, maps) -> np.ndarray:
        hc = self.b.households
        rad = max(1, int(round(float(hc.search_radius_km) * 1000.0 / self.grid.dx)))
        k_near = int(hc.candidates_near)
        k_far = int(hc.candidates_far)
        out = []
        if center >= 0:
            r0, c0 = divmod(int(center), self.nx)
            dr = r.integers(-rad, rad + 1, size=k_near)
            dc = r.integers(-rad, rad + 1, size=k_near)
            rr = np.clip(r0 + dr, 0, self.ny - 1)
            cc = np.clip(c0 + dc, 0, self.nx - 1)
            out.append(rr * self.nx + cc)
        cdf = maps["far_cdf"]
        if cdf is not None:
            out.append(np.minimum(np.searchsorted(cdf, r.random(k_far), side="right"), self.n_cells - 1))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)

    def _far_cdf(self, maps):
        """マップ全体から候補を選ぶときの重み（累積分布）：働き手2人の世帯が見込める収穫と、人の集まり。"""
        score = np.minimum(2.0 / np.maximum(maps["labor"], 1e-12), maps["A"]) * maps["yield"] + maps["fish"] * 2.0
        score = np.where(self.passable, score, 0.0) + np.where(self.passable, maps["social"], 0.0)
        tot = score.sum()
        if tot <= 0:
            return None
        return np.cumsum(score) / tot

    def _choose(self, r, h: int, current: int, center: int, maps, allow_outside: bool, income_now=None,
                inside: bool = True):
        """世帯 h の住む場所をロジットで選ぶ。戻り値：(マス, 仕事)。マス = -1 はマップの外へ出る。

        inside：今マップの中にいる（独立した世帯・見直す世帯）なら True。マップの外へ出る選択肢に引っ越しの費用がかかる。
        外から来る世帯なら False。マップの中のマスを選ぶと引っ越しの費用がかかる。
        """
        hc = self.b.households
        cands = self._sample_candidates(r, center, maps)
        if current >= 0:
            cands = np.unique(np.concatenate([cands, [current]]))
        ids = np.full(cands.size, h)
        L = np.full(cands.size, float(self._labor(np.array([h]))[0]))
        inc = None if income_now is None else np.full(cands.size, income_now)
        val, occ = self._utility(ids, cands, L, maps, current, inc, move_cost=(current >= 0 or not inside))
        vals = list(val)
        if allow_outside:
            mc = float(self.b.outside.migration.moving_cost_utility)
            vals.append(self._outside_utility() - (mc if inside else 0.0))
        v = np.array(vals) / float(hc.utility.logit_scale)
        if not np.isfinite(v).any():
            return -1, OCC_NONE
        v = v - v[np.isfinite(v)].max()
        pr = np.where(np.isfinite(v), np.exp(v), 0.0)
        pr /= pr.sum()
        pick = int(r.choice(pr.size, p=pr))
        if pick >= cands.size:
            return -1, OCC_NONE
        return int(cands[pick]), int(occ[pick])

    def _settle(self, h: int, cell: int, occ: int, area=None) -> None:
        """世帯 h が cell に住み、仕事を始める。農業なら近くの田畑を手に入れる。area は田畑の面積（まとめて計算したもの）。"""
        old = self.hh.cell[h]
        self.hh.cell[h] = cell
        self.hh.occ[h] = occ
        self.hh.shop_kind[h] = -1
        if old >= 0 and old != cell:
            self._release(h, cell)
        if occ == OCC_FARM:
            self._acquire_land(h, area)
        else:
            self._release(h, None)

    def _acquire_land(self, h: int, area=None) -> None:
        """世帯 h が働き手で耕せる面積まで、近くの田畑を手に入れる。
        優先順：持ち主のいない田畑（すぐ耕せる）→ 借りられる田畑（小作）→ 未開の土地を開墾（1年に上限まで）。
        同じ種類の中では、見込みの収穫が多く、家に近い田畑を選ぶ。"""
        f = self.b.farming
        land = self.land
        home = int(self.hh.cell[h])
        L = float(self._labor(np.array([h]))[0])
        if area is None:
            area = self._farm_area(self._res_count())
        mine = np.flatnonzero(land.cultivator == h)
        have = float((self.labor_per_ha[mine] * area[mine]).sum())
        if have >= L:
            return
        win = self._window(home, self.r_cult)
        win = win[self._dist_km(win, np.full(win.size, home)) <= float(f.cultivation_radius_km)]
        win = win[area[win] > 0]
        d = self._dist_km(win, np.full(win.size, home))
        score = self.pot[win] / (1.0 + float(f.distance_labor_loss_per_km) * d)
        groups = [
            (land.use[win] > 0) & (land.owner[win] < 0) & (land.cultivator[win] < 0),
            (land.use[win] > 0) & (land.owner[win] >= 0) & (land.cultivator[win] < 0),
            (land.use[win] == USE_WILD) & self.arable[win] & (land.owner[win] < 0),
        ]
        reclaimed = 0.0
        for gi, g in enumerate(groups):
            cand = win[g]
            sc = score[g]
            order = np.argsort(-sc, kind="stable")
            for c in cand[order]:
                if have >= L:
                    return
                if land.cultivator[c] >= 0:
                    continue
                if gi == 0:
                    land.owner[c] = h
                elif gi == 2:
                    if reclaimed >= float(f.reclaim_per_year_max_ha):
                        break
                    land.owner[c] = h
                    land.use[c] = self.crop[c]
                    land.reclaim_left[c] = self.reclaim_py[c] * area[c]
                    reclaimed += area[c]
                land.cultivator[c] = h
                have += self.labor_per_ha[c] * area[c]

    def _rebalance_farms(self) -> None:
        """農家ごとに、働き手で耕せる面積に合わせて田畑を増やす・手放す。"""
        hh = self.hh
        ids = hh.ids()
        farmers = ids[hh.occ[ids] == OCC_FARM]
        if farmers.size == 0:
            return
        res = self._res_count()
        area = self._farm_area(res)
        L = self._labor(farmers)
        plots = np.flatnonzero(self.land.cultivator >= 0)
        need = np.bincount(self.land.cultivator[plots], weights=self.labor_per_ha[plots] * area[plots], minlength=hh.n)
        ratio = float(self.b.farming.release_surplus_ratio)
        for h, l in zip(farmers, L):
            if need[h] > ratio * max(l, 1e-9):
                mine = np.flatnonzero(self.land.cultivator == h)
                order = np.argsort(self.pot[mine] - 1e-3 * self._dist_km(mine, np.full(mine.size, hh.cell[h])), kind="stable")
                tot = need[h]
                for c in mine[order]:
                    if tot <= l:
                        break
                    self.land.cultivator[c] = -1
                    tot -= self.labor_per_ha[c] * area[c]
            elif need[h] < l:
                self._acquire_land(h, area)

    def _reviews(self, dt: float) -> None:
        """一部の世帯が、住む場所と仕事を見直す（SPEC 9.1）。仕事がない世帯は必ず見直す。"""
        hh = self.hh
        r = self.rng["review"]
        ids = hh.ids()
        if ids.size == 0:
            return
        p = 1.0 - np.exp(-float(self.b.households.review_prob_per_year) * dt)
        draws = r.random(ids.size)
        who = ids[((draws < p) & (hh.occ[ids] != OCC_SHOP)) | (hh.occ[ids] == OCC_NONE)]
        maps = self._opportunity_maps()
        for h in who:
            cur = int(hh.cell[h])
            cell, occ = self._choose(r, int(h), cur, cur, maps, True, income_now=hh.income[h] if hh.occ[h] != OCC_NONE else None)
            if cell < 0:
                self._remove_household(int(h))
                self.stats_year["emigrants"] += 1
                continue
            if cell == cur and hh.occ[h] != OCC_NONE:
                continue
            self._settle(int(h), cell, occ, maps["area"])
            maps["res"][cur] -= 1
            maps["res"][cell] += 1

    def _immigration(self, dt: float) -> None:
        """外部の大都市から、移住を考える世帯がやって来て、来るかどうかを自分で決める（ユーザー合意：ウォームアップの始まり方）。"""
        o = self.b.outside
        d = self.b.demography.migrant_household
        r = self.rng["migration"]
        if self.conn_cells.size == 0 or not self.trees:
            return
        maps = None
        base = np.array(d.ages, dtype=np.int64)
        spread = int(d.spread_years)
        for k, c in enumerate(self.conn_cells):
            lam = float(o.migration.potential_households_per_year_per_100k) * self.conn_pop[k] / 1e5 * dt
            m = int(r.poisson(lam))
            for _ in range(m):
                ages = np.maximum(base + r.integers(-spread, spread + 1, size=base.size), 0)
                if maps is None:
                    maps = self._opportunity_maps()
                # まず接続点にいることにして加え、選んだマスへ移す（外へ戻るなら消す）
                h = int(self.hh.add(np.array([int(c)]), ages[None, :].astype(np.int16), np.array([float(o.outside_household.money_yen)]))[0])
                cell, occ = self._choose(r, h, -1, int(c), maps, True, inside=False)
                if cell < 0:
                    self.hh.alive[h] = False
                    continue
                self._settle(h, cell, occ, maps["area"])
                maps["res"][cell] += 1
                self.stats_year["immigrants"] += 1

    # ------------------------------------------------------------------
    # 人口
    # ------------------------------------------------------------------
    def _demography(self) -> None:
        dc = self.b.demography
        hh = self.hh
        r = self.rng["demography"]
        ids = hh.ids()
        if ids.size == 0:
            return
        ages = hh.ages[ids]
        present = ages >= 0
        ages = np.where(present, ages + 1, -1)
        # 死亡
        lower = np.array(dc.death.age_lower, dtype=np.int64)
        prob = np.array(dc.death.prob_per_year, dtype=np.float64)
        band = np.searchsorted(lower, np.maximum(ages, 0), side="right") - 1
        short = np.maximum(0.0, 1.0 - hh.food_ratio[ids])[:, None]
        q = np.clip(prob[band] * (1.0 + float(dc.death.famine_multiplier_per_shortfall) * short), 0.0, 1.0)
        die = (r.random(ages.shape) < q) & present
        self.stats_year["deaths"] += int(die.sum())
        ages = np.where(die, -1, ages)
        # 出生
        bc = dc.birth
        fertile = ((ages >= int(bc.min_age)) & (ages <= int(bc.max_age))).sum(axis=1)
        pb = float(bc.per_adult_per_year) * np.where(hh.food_ratio[ids] < 1.0, float(bc.famine_multiplier), 1.0)
        births = r.binomial(fertile, np.clip(pb, 0.0, 1.0))
        for i in np.flatnonzero(births > 0):
            free = np.flatnonzero(ages[i] < 0)
            nb = min(int(births[i]), free.size)
            ages[i, free[:nb]] = 0
            self.stats_year["births"] += nb
        hh.ages[ids] = ages
        # 誰もいなくなった世帯
        empty = ids[(ages >= 0).sum(axis=1) == 0]
        for h in empty:
            self._remove_household(int(h))
        # 独立
        self._household_formation()

    def _household_formation(self) -> None:
        fc = self.b.demography.household_formation
        hh = self.hh
        r = self.rng["formation"]
        ids = hh.ids()
        leavers = []
        for h in ids:
            a = hh.ages[h]
            members = np.flatnonzero(a >= 0)
            if members.size < int(fc.min_members_to_leave):
                continue
            young = members[(a[members] >= int(fc.min_age)) & (a[members] <= int(fc.max_age))]
            if young.size == 0:
                continue
            # 跡継ぎ（年長の若者）は残る
            heir = young[np.argmax(a[young])]
            for m in young:
                if m == heir:
                    continue
                if r.random() < float(fc.prob_per_year):
                    leavers.append((int(h), int(a[m])))
                    a[m] = -1
        if not leavers:
            return
        groups = []
        if bool(fc.pair_up):
            for i in range(0, len(leavers) - 1, 2):
                groups.append([leavers[i], leavers[i + 1]])
            if len(leavers) % 2:
                groups.append([leavers[-1]])
        else:
            groups = [[x] for x in leavers]
        share = float(fc.money_share_from_parent)
        maps = self._opportunity_maps()
        for g in groups:
            parent = g[0][0]
            money = 0.0
            for p, _ in g:
                take = share * hh.money[p]
                hh.money[p] -= take
                money += take
            ages = np.array([[age for _, age in g]], dtype=np.int16)
            h = int(hh.add(np.array([hh.cell[parent]]), ages, np.array([money]))[0])
            cell, occ = self._choose(r, h, -1, int(hh.cell[parent]), maps, True)
            if cell < 0:
                hh.alive[h] = False
                self.stats_year["emigrants"] += 1
                continue
            hh.cell[h] = cell
            self._settle(h, cell, occ, maps["area"])
            maps["res"][cell] += 1
            self.stats_year["new_households"] += 1

    # ------------------------------------------------------------------
    # 土地市場
    # ------------------------------------------------------------------
    def _land_market(self, years: float) -> None:
        lc = self.b.land.residential_rent
        res = self._res_count()
        occ = res / self.cap_hh
        rent = self.land.res_rent * np.exp(float(lc.adjust_rate) * years * (occ - float(lc.target_occupancy)))
        self.land.res_rent = np.clip(np.where(res > 0, rent, self.land.res_rent), float(lc.floor_yen), float(lc.max_yen))
        # 小作料の割合：借りたい面積（田畑が足りない農家）と、貸しに出ている面積の比で調整
        tc = self.b.farming.tenancy
        area = self._farm_area(res)
        supply = float(area[(self.land.use > 0) & (self.land.owner >= 0) & (self.land.cultivator < 0)].sum())
        ids = self.hh.ids()
        farmers = ids[self.hh.occ[ids] == OCC_FARM]
        L = self._labor(farmers)
        plots = np.flatnonzero(self.land.cultivator >= 0)
        have = np.bincount(self.land.cultivator[plots], weights=area[plots], minlength=self.hh.n)
        demand = float(np.maximum(L / 2.0 - have[farmers], 0.0).sum())
        x = (demand - supply) / max(demand + supply, 1e-9)
        self.tenancy_share = float(np.clip(self.tenancy_share * np.exp(float(tc.adjust_rate) * years * x),
                                           float(tc.min_share), float(tc.max_share)))
        # 地価 = 年間の地代 ÷ 金利
        p = self.p_rice
        farm_rent = self.tenancy_share * self.pot * area * p * (self.land.use > 0)
        total = farm_rent + self.land.res_rent * res
        self.land.land_price = total / float(self.b.outside.interest_rate_per_year) / self.area_ha

    # ------------------------------------------------------------------
    # ステップ
    # ------------------------------------------------------------------
    def step_year(self, dT: float, pf: float) -> dict:
        """ウォームアップの1年。"""
        self.stats_year = self._empty_stats()
        self._transport_update(1.0)
        self._harvest_and_income(dT, pf)
        self._commerce_exit()
        self._demography()
        self._immigration(1.0)
        self._reviews(1.0)
        self._rebalance_farms()
        self._commerce_entry(1.0)
        self._land_market(1.0)
        return self.stats_year

    def step_month(self, month: int, dT: float, pf: float) -> dict:
        """本番の1か月。1年に1回の出来事は決まった月に行う。"""
        f = self.b.farming
        interval = int(self.b.transport.roads.update_interval_months)
        if month == 1:
            self.stats_year = self._empty_stats()
        self.months_since_transport += 1
        if self.months_since_transport >= interval:
            self._transport_update(interval / 12.0)
        if month == 1:
            self._demography()
        if month == int(f.harvest_month):
            self._harvest_and_income(dT, pf)
            self._commerce_exit()
            self._rebalance_farms()
        if month == 12:
            self._land_market(1.0)
        dt = 1.0 / 12.0
        self._immigration(dt)
        self._reviews(dt)
        self._commerce_entry(dt)
        return self.stats_year

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, np.ndarray]:
        hh = self.hh
        ids = hh.ids()
        shape = (self.ny, self.nx)
        pop = np.bincount(hh.cell[ids], weights=hh.members(ids), minlength=self.n_cells)
        res = self._res_count()
        shops = self._shop_counts()
        area = self._farm_area(res)
        use = self.land.use.copy()
        lu = np.where(use > 0, use, 0).astype(np.int8)
        lu = np.where((res > 0) & (area < 0.5 * self.area_ha), 3, lu)
        lu = np.where(shops.sum(axis=0) > 0, 4, lu).astype(np.int8)
        return {
            "population": pop.reshape(shape).astype(np.int32),
            "households": res.reshape(shape).astype(np.int32),
            "land_use": lu.reshape(shape),
            "road": self.land.road.reshape(shape).astype(np.float32),
            "land_price": self.land.land_price.reshape(shape).astype(np.float32),
            "shops_daily": shops[SHOP_DAILY].reshape(shape).astype(np.int16),
            "shops_specialty": shops[SHOP_SPECIALTY].reshape(shape).astype(np.int16),
            "inns": shops[SHOP_INN].reshape(shape).astype(np.int16),
        }

    def stats(self) -> dict[str, float]:
        hh = self.hh
        ids = hh.ids()
        occ = hh.occ[ids]
        out = {
            "population": float(hh.members(ids).sum()),
            "households": float(ids.size),
            "farmers": float((occ == OCC_FARM).sum()),
            "fishers": float((occ == OCC_FISH).sum()),
            "shops": float((occ == OCC_SHOP).sum()),
            "farmland_ha": float((self.land.use > 0).sum() * self.area_ha),
            "tenancy_share": self.tenancy_share,
            "rice_export": float(self.rice_export),
        }
        out.update({k: float(v) for k, v in self.stats_year.items()})
        return out
