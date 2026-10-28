"""鉱業の会社と、鉱山に出資する投資家（SPEC 9.2・9.4）。

鉱山は上から置かない。投資家が毎年、開かれていない鉱床の事業の見込みを金利と比べ、出資すると判断したときにだけ会社が生まれる。
会社は自分の利益だけで、雇う人数・賃金・設備を決め、埋蔵量が尽きるか赤字が続けば撤退する。
1つの鉱床に会社は1社まで。状態は鉱床ごとの配列で持つ（配列の構造体）。
"""

from __future__ import annotations

import numpy as np

from .. import kernels
from .state import OCC_NONE, OCC_WAGE

EDGE_LABELS = {"north": "北", "south": "南", "east": "東", "west": "西"}
KIND_LABELS = {"river": "川の上流", "sea": "海", "pass": "峠", "plain": "平地"}


class Mining:
    def __init__(self, soc, info: dict, arrays: dict, mcfg, ocfg):
        self.cfg = mcfg
        self.ocfg = ocfg
        deps = info["deposits"]
        n = len(deps)
        self.n = n
        self.mineral = [d["mineral"] for d in deps]
        self.label = [d["label"] for d in deps]
        self.grade = np.array([d["grade"] for d in deps], dtype=np.float64)
        self.reserve0 = np.array([d["reserve_t"] for d in deps], dtype=np.float64)
        self.left = self.reserve0.copy()
        self.depth_top = np.array([d["depth_top_m"] for d in deps], dtype=np.float64)
        center = np.array([d["row"] * soc.nx + d["col"] for d in deps], dtype=np.int64)
        self.center = center
        # 鉱床の広がり：同じ鉱物の埋蔵量の格子のマスを、最も近い鉱床の中心に割り当てる
        self.cells: list[np.ndarray] = [np.array([c], dtype=np.int64) for c in center]
        for m in set(self.mineral):
            ras = arrays[f"mineral_{m}_t"].ravel()
            cells = np.flatnonzero(ras > 0)
            ks = [k for k in range(n) if self.mineral[k] == m]
            if cells.size == 0:
                continue
            d2 = np.stack([(soc.rows[cells] - soc.rows[center[k]]) ** 2 + (soc.cols[cells] - soc.cols[center[k]]) ** 2 for k in ks])
            owner = np.argmin(d2, axis=0)
            for j, k in enumerate(ks):
                mine = cells[owner == j]
                if mine.size:
                    self.cells[k] = mine
        c = mcfg.common
        self.lref = float(c.ref_workers_per_mt) * (self.reserve0 / 1e6) ** float(c.reserve_exponent)

        self.active = np.zeros(n, dtype=bool)
        self.site = np.full(n, -1, dtype=np.int64)
        self.capital = np.zeros(n)          # 働き手の分の設備 [円]（固定の開削費は含まない）
        self.wage = np.zeros(n)             # 日当 [円/日]
        self.target = np.zeros(n)           # 雇いたい働き手の数
        self.loss_years = np.zeros(n, dtype=np.int64)
        self.output = np.zeros(n)           # 去年の採掘量 [t]
        self.profit = np.zeros(n)
        self.investor = [""] * n
        self.local_holders: list[dict] = [{} for _ in range(n)]   # 地元の出資者：世帯 → 持ち分の割合
        self.company = np.full(n, -1, dtype=np.int64)
        self.next_company = 0
        self.commute: dict[int, np.ndarray] = {}
        self.events: list[dict] = []
        self.productivity = 1.0             # 外部の生産性（Society が毎年入れる）

    # ------------------------------------------------------------------
    def mineral_cfg(self, d: int):
        return self.cfg.minerals[self.mineral[d]]

    def depth(self, d: int) -> float:
        done = 1.0 - self.left[d] / max(self.reserve0[d], 1e-12)
        return float(self.depth_top[d] + done * float(self.cfg.common.thickness_m))

    def _q_eff(self, d: int) -> float:
        c = self.cfg.common
        tech = self.productivity ** float(c.productivity_follows_outside)
        return float(self.mineral_cfg(d).output_t_per_worker) * tech * np.exp(-self.depth(d) / float(c.depth_scale_m))

    def produce(self, d: int, L: float) -> float:
        a = float(self.cfg.common.labor_exponent)
        if L <= 0.0:
            return 0.0
        return self._q_eff(d) * L ** a * self.lref[d] ** (1.0 - a)

    def net_per_t(self, d: int, site: int, prices: dict, freight_to_conn: np.ndarray) -> float:
        mc = self.mineral_cfg(d)
        rec = float(mc.recovery)
        value = prices[self.mineral[d]] * self.grade[d] / float(mc.reference_grade) * rec
        ship = self.grade[d] * rec if bool(mc.smelted_on_site) else 1.0
        fr = freight_to_conn[site] * 1000.0 * ship
        if not np.isfinite(fr):
            return -np.inf
        return float(value - float(mc.opex_yen_per_t) - fr)

    def l_star(self, d: int, net: float, wage_year: float) -> float:
        a = float(self.cfg.common.labor_exponent)
        if net <= 0.0 or wage_year <= 0.0:
            return 0.0
        return float(self.lref[d] * (a * self._q_eff(d) * net / wage_year) ** (1.0 / (1.0 - a)))

    def fixed_capital(self, d: int) -> float:
        c = self.cfg.common
        return float(c.shaft_base_yen) + float(c.shaft_yen_per_m) * self.depth(d)

    def best_site(self, soc, d: int) -> int:
        cells = self.cells[d]
        cells = cells[soc.passable[cells]]
        if cells.size == 0:
            return -1
        fr = soc.freight_to_conn[cells]
        if not np.isfinite(fr).any():
            return -1
        return int(cells[np.argmin(fr)])

    def expected_return(self, soc, d: int, site: int, wage_day: float) -> tuple[float, float]:
        """利益が最大の規模で開いた場合の (期待収益率 [1/年], 働き手の数)。"""
        c = self.cfg.common
        days = float(self.ocfg.work_days_per_year)
        net = self.net_per_t(d, site, soc.prices, soc.freight_to_conn)
        L = self.l_star(d, net, wage_day * days)
        if L <= 0.0:
            return -np.inf, 0.0
        Q = self.produce(d, L)
        K = float(c.capital_yen_per_worker) * L + self.fixed_capital(d)
        prof = net * Q - wage_day * days * L
        ret = (prof - float(c.depreciation_per_year) * K) / K - Q / max(self.left[d], 1e-12)
        return float(ret), L

    # ------------------------------------------------------------------
    def employed(self, soc) -> np.ndarray:
        """鉱床ごとの、雇われている働き手の数。"""
        hh = soc.hh
        ids = hh.ids()
        w = ids[hh.occ[ids] == OCC_WAGE]
        if w.size == 0 or self.n == 0:
            return np.zeros(self.n)
        return np.bincount(hh.employer[w], weights=soc._labor(w), minlength=self.n)[: self.n]

    def refresh_commute(self, soc) -> None:
        cut = float(self.cfg.labor.commute_max_hours)
        self.commute = {int(d): soc.transport.person_tree(int(self.site[d]), cut)[0] for d in np.flatnonzero(self.active)}

    def job_maps(self, soc, vacancy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """マスごとの、通える求人の中で最もよい手取り [円/年/働き手]（通勤の手間を引いた値）と、その会社の鉱床。"""
        days = float(self.ocfg.work_days_per_year)
        hours = float(self.ocfg.work_hours_per_day)
        best = np.full(soc.n_cells, -np.inf)
        firm = np.full(soc.n_cells, -1, dtype=np.int64)
        for d, t in self.commute.items():
            if not self.active[d] or vacancy[d] <= 0.0:
                continue
            net = np.where(np.isfinite(t), self.wage[d] * days * (1.0 - 2.0 * t / hours), -np.inf)
            better = net > best
            best = np.where(better, net, best)
            firm = np.where(better, d, firm)
        return best, firm

    def ore_traffic(self, soc, tr: np.ndarray) -> None:
        """鉱石（製錬したものは地金）を外部とつながる地点まで運ぶ荷車の往復を、通行量に加える。"""
        load = float(soc.b.transport.cart.load_kg)
        for d in np.flatnonzero(self.active):
            s = int(self.site[d])
            mc = self.mineral_cfg(d)
            ship = self.grade[d] * float(mc.recovery) if bool(mc.smelted_on_site) else 1.0
            trips = 2.0 * self.output[d] * ship * 1000.0 / load
            if trips <= 0.0 or not soc.conn_cells.size:
                continue
            cc = int(soc.conn_cells[soc.best_conn[s]])
            kernels.add_path_traffic(soc.trees[cc][1], np.array([s], dtype=np.int64), np.array([trips]), tr)

    # ------------------------------------------------------------------
    def accounting(self, soc, L_h: np.ndarray, dt_years: float = 1.0) -> np.ndarray:
        """1年分の採掘と収支。戻り値は世帯ごとの賃金と配当 [円/年]。"""
        hh = soc.hh
        out = np.zeros(hh.n)
        days = float(self.ocfg.work_days_per_year)
        c = self.cfg.common
        ids = hh.ids()
        workers = ids[hh.occ[ids] == OCC_WAGE]
        emp = np.bincount(hh.employer[workers], weights=L_h[workers], minlength=self.n)[: self.n] if workers.size else np.zeros(self.n)
        for d in np.flatnonzero(self.active):
            s = int(self.site[d])
            Q = min(self.produce(d, emp[d]), self.left[d])
            net = self.net_per_t(d, s, soc.prices, soc.freight_to_conn)
            self.left[d] -= Q
            self.output[d] = Q
            wages = self.wage[d] * days * emp[d]
            mine = workers[hh.employer[workers] == d]
            out[mine] += self.wage[d] * days * L_h[mine]
            # 坑口の土地の地代（田畑として貸した場合と同じ額）は持ち主に入る
            rent = soc.tenancy_share * soc.pot[s] * soc.area_ha * soc.p_rice[s]
            owner = int(soc.land.owner[s])
            if owner >= 0 and hh.alive[owner]:
                out[owner] += rent
            K = self.capital[d] + self.fixed_capital(d)
            prof = net * Q - wages - float(c.depreciation_per_year) * K - rent
            self.profit[d] = prof
            # 撤退の判断は見込みで行う：今の価格と賃金で、目標の人数を雇えたときの利益（開削費は埋没費用なので含めない）。
            # 実際の利益で判断すると、人が集まる前の最初の年の赤字だけで撤退してしまう
            tgt = self.target[d]
            expect = net * min(self.produce(d, tgt), self.left[d]) - self.wage[d] * days * tgt - float(c.depreciation_per_year) * self.capital[d] - rent
            self.loss_years[d] = self.loss_years[d] + 1 if expect < 0.0 else 0
            # 配当：地元の出資者の持ち分だけ、フィールドの中の世帯に入る（外部の投資家の分は外へ出る）
            if prof > 0.0:
                for h, share in self.local_holders[d].items():
                    if hh.alive[h]:
                        out[h] += share * prof
        return out

    def adjust(self, soc, year: int, month: int) -> None:
        """撤退・雇う人数・賃金・設備の見直し（1年に1回）。"""
        c = self.cfg.common
        lc = self.cfg.labor
        days = float(self.ocfg.work_days_per_year)
        emp = self.employed(soc)
        for d in np.flatnonzero(self.active):
            if self.left[d] <= 0.0 or self.loss_years[d] >= int(c.exit_loss_years):
                self._close(soc, int(d), year, month, "exhausted" if self.left[d] <= 0.0 else "unprofitable")
                continue
            s = int(self.site[d])
            net = self.net_per_t(d, s, soc.prices, soc.freight_to_conn)
            Ls = self.l_star(d, net, self.wage[d] * days)
            kpw = float(c.capital_yen_per_worker)
            if Ls > self.capital[d] / kpw and emp[d] >= 0.8 * self.target[d]:   # 今の設備で人が集まっているなら設備を増やす
                self.capital[d] = min(self.capital[d] * (1.0 + float(c.max_expansion_per_year)), Ls * kpw)
            self.target[d] = min(Ls, self.capital[d] / kpw)
            short = (self.target[d] - emp[d]) / max(self.target[d], 1e-9)
            if short > 0.0:
                self.wage[d] *= np.exp(float(lc.adjust_up_per_year) * short)
            else:
                self.wage[d] *= np.exp(-float(lc.adjust_down_per_year))
            self.wage[d] = max(self.wage[d], float(lc.min_wage_ratio) * soc.ext.wage_yen_per_day)
            if emp[d] > float(lc.layoff_excess_ratio) * self.target[d]:
                self._lay_off(soc, int(d), emp[d] - self.target[d])

    def _lay_off(self, soc, d: int, excess: float) -> None:
        hh = soc.hh
        ids = hh.ids()
        mine = ids[(hh.occ[ids] == OCC_WAGE) & (hh.employer[ids] == d)][::-1]   # 後から雇った世帯から
        L = soc._labor(mine)
        for h, l in zip(mine, L):
            if excess <= 0.0:
                break
            hh.occ[h] = OCC_NONE
            hh.employer[h] = -1
            excess -= l

    def _close(self, soc, d: int, year: int, month: int, reason: str) -> None:
        hh = soc.hh
        ids = hh.ids()
        mine = ids[(hh.occ[ids] == OCC_WAGE) & (hh.employer[ids] == d)]
        hh.occ[mine] = OCC_NONE
        hh.employer[mine] = -1
        self.active[d] = False
        soc.land.mine[self.site[d]] = False
        self.events.append({"year": year, "month": month, "event": "mine_closed", "reason": reason,
                            "mineral": self.mineral[d], "label": self.label[d], "company": int(self.company[d]),
                            "row": int(soc.rows[self.site[d]]), "col": int(soc.cols[self.site[d]]),
                            "left_t": float(self.left[d]), "workers_laid_off": int(mine.size)})
        self.commute.pop(d, None)
        self.capital[d] = 0.0
        self.target[d] = 0.0
        self.output[d] = 0.0
        self.local_holders[d] = {}

    def invest(self, soc, r, year: int, month: int) -> None:
        """投資家が、開かれていない鉱床の事業に出資するかを決める（1年に1回）。"""
        if not soc.trees:
            return
        ic = self.cfg.investors
        c = self.cfg.common
        ext = soc.ext
        wage0 = ext.wage_yen_per_day * (1.0 + float(self.cfg.labor.start_wage_premium))
        # 出資できる資金：外部の大都市ごとの1年分の資金と、地元の裕福な世帯の蓄えの余り
        funds = [float(ic.fund_per_capita_yen) * p * ext.productivity for p in ext.city_pop]
        ids = soc.hh.ids()
        spare = np.maximum(soc.hh.money[ids] - float(ic.local_min_money_yen), 0.0)
        local = float(spare.sum())
        draws = r.random((self.n, 3))   # 鉱床ごとに同じ回数だけ引く（条件で乱数列がずれないように）
        for d in range(self.n):
            if self.active[d] or self.left[d] <= 0.0:
                continue
            if draws[d, 0] >= float(ic.consider_prob_per_year):
                continue
            site = self.best_site(soc, d)
            if site < 0:
                continue
            ret, L = self.expected_return(soc, d, site, wage0)
            if not np.isfinite(ret):
                continue
            ge = float(self.ocfg.goods[self.mineral[d]].cycle_elasticity)
            hurdle = ext.interest + float(ic.risk_premium) + float(ic.cycle_risk_per_amplitude) * soc.external.amplitude * ge
            p = 1.0 / (1.0 + np.exp(-(ret - hurdle) / float(ic.logit_scale)))
            if draws[d, 1] >= p:
                continue
            K = float(c.start_fraction) * float(c.capital_yen_per_worker) * L
            need = K + self.fixed_capital(d)
            pool = [f for f in funds] + [local]
            ok = np.array([f >= need for f in pool])
            if not ok.any():
                continue
            w = np.where(ok, np.array(pool), 0.0)
            pick = int(np.searchsorted(np.cumsum(w) / w.sum(), draws[d, 2], side="right"))
            pick = min(pick, len(pool) - 1)
            if pick < len(funds):
                funds[pick] -= need
                city = soc.external_cities[pick]
                who = f"{EDGE_LABELS.get(city['edge'], city['edge'])}の{KIND_LABELS.get(city['kind'], city['kind'])}の都市の投資家"
                holders = {}
            else:
                share = spare / max(local, 1e-12)
                take = share * need
                soc.hh.money[ids] -= take
                spare = spare - take
                local -= need
                holders = {int(h): float(s) for h, s in zip(ids, share) if s > 0.0}
                who = "地元の投資家"
            self._open(soc, d, site, K, wage0, who, holders, year, month)

    def _open(self, soc, d: int, site: int, K: float, wage: float, who: str, holders: dict, year: int, month: int) -> None:
        self.active[d] = True
        self.site[d] = site
        self.capital[d] = K
        self.wage[d] = wage
        self.target[d] = K / float(self.cfg.common.capital_yen_per_worker)
        self.loss_years[d] = 0
        self.profit[d] = 0.0
        self.investor[d] = who
        self.local_holders[d] = holders
        self.company[d] = self.next_company
        self.next_company += 1
        soc.land.mine[site] = True
        soc.land.cultivator[site] = -1
        cut = float(self.cfg.labor.commute_max_hours)
        self.commute[d] = soc.transport.person_tree(site, cut)[0]
        self.events.append({"year": year, "month": month, "event": "mine_opened", "mineral": self.mineral[d],
                            "label": self.label[d], "company": int(self.company[d]), "investor": who,
                            "row": int(soc.rows[site]), "col": int(soc.cols[site]), "workers_target": float(self.target[d])})
