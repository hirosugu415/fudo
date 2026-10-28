"""社会の状態。配列の構造体（属性ごとの NumPy 配列）で持つ（SPEC 第2章）。"""

from __future__ import annotations

import numpy as np

# 仕事
OCC_NONE = -1     # 仕事がない（独立したばかり・店をやめた直後など。次の見直しで決める）
OCC_FARM = 0      # 農業（自作・小作）
OCC_FISH = 1      # 漁業
OCC_SHOP = 2      # 商い（店の種類は shop_kind）
OCC_WAGE = 3      # 雇われて働く（鉱山。勤め先は employer）

# 店の種類
SHOP_DAILY = 0
SHOP_SPECIALTY = 1
SHOP_INN = 2
SHOP_KINDS = ("daily", "specialty", "inn")

# 土地利用（マスごと）
USE_WILD = 0      # 森林・草地・荒れ地
USE_PADDY = 1     # 水田
USE_FIELD = 2     # 畑


class Households:
    """世帯。番号（添字）は世帯が消えても再利用しない（決定性と追跡のため）。"""

    def __init__(self, max_members: int, capacity: int = 1024):
        self.max_members = max_members
        self.n = 0
        self._alloc(capacity)

    def _alloc(self, cap: int) -> None:
        def grow(name, shape_tail, dtype, fill):
            old = getattr(self, name, None)
            arr = np.full((cap,) + shape_tail, fill, dtype=dtype)
            if old is not None:
                arr[: old.shape[0]] = old
            setattr(self, name, arr)

        m = self.max_members
        grow("alive", (), np.bool_, False)
        grow("cell", (), np.int32, -1)
        grow("ages", (m,), np.int16, -1)
        grow("money", (), np.float64, 0.0)
        grow("occ", (), np.int8, OCC_NONE)
        grow("shop_kind", (), np.int8, -1)
        grow("income", (), np.float64, 0.0)
        grow("food_ratio", (), np.float64, 1.0)
        grow("loss_years", (), np.int16, 0)
        grow("employer", (), np.int32, -1)
        self.cap = cap

    def add(self, cells: np.ndarray, ages: np.ndarray, money: np.ndarray) -> np.ndarray:
        k = int(cells.size)
        while self.n + k > self.cap:
            self._alloc(self.cap * 2)
        ids = np.arange(self.n, self.n + k, dtype=np.int64)
        self.alive[ids] = True
        self.cell[ids] = cells
        self.ages[ids] = -1
        self.ages[ids, : ages.shape[1]] = ages
        self.money[ids] = money
        self.occ[ids] = OCC_NONE
        self.shop_kind[ids] = -1
        self.income[ids] = 0.0
        self.food_ratio[ids] = 1.0
        self.loss_years[ids] = 0
        self.employer[ids] = -1
        self.n += k
        return ids

    def ids(self) -> np.ndarray:
        """生きている世帯の番号（昇順）。"""
        return np.flatnonzero(self.alive[: self.n])

    def members(self, ids=None) -> np.ndarray:
        a = self.ages[: self.n] if ids is None else self.ages[ids]
        return (a >= 0).sum(axis=1)


class Land:
    """マスごとの土地の状態。"""

    def __init__(self, n: int, initial_rent: float):
        self.use = np.zeros(n, dtype=np.int8)
        self.owner = np.full(n, -1, dtype=np.int32)        # 所有する世帯（開墾した世帯）
        self.cultivator = np.full(n, -1, dtype=np.int32)   # 耕している世帯（自作なら所有者、小作なら借りた世帯）
        self.reclaim_left = np.zeros(n, dtype=np.float64)  # 開墾の残りの手間 [人年]（> 0 なら開墾中）
        self.res_rent = np.full(n, initial_rent, dtype=np.float64)   # 住む場所の地代 [円/世帯/年]
        self.road = np.zeros(n, dtype=np.float64)          # 道の状態（0〜1）
        self.fish_stock = np.zeros(n, dtype=np.float64)    # 漁場の資源量 [t]
        self.land_price = np.zeros(n, dtype=np.float64)    # 地価 [円/ha]
        self.mine = np.zeros(n, dtype=bool)                # 鉱山の坑口
