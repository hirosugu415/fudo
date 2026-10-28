"""格子。行（row）は北から南、列（col）は西から東へ数える。セル番号は row * nx + col。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 8近傍の順番（固定）：北、北東、東、南東、南、南西、西、北西
NEIGH_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
NEIGH_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
NEIGH_DIST = np.array([1.0, np.sqrt(2.0)] * 4, dtype=np.float64)  # セル幅を1としたときの距離

# マップ端の番号：北、東、南、西
EDGES = ("north", "east", "south", "west")
EDGE_LABELS = {"north": "北", "east": "東", "south": "南", "west": "西"}


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float  # m

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx

    @classmethod
    def from_config(cls, frame) -> "Grid":
        g = frame.grid
        return cls(int(g.nx), int(g.ny), float(g.cell_size_m))

    def edge_mask(self, edge: str) -> np.ndarray:
        m = np.zeros((self.ny, self.nx), dtype=bool)
        if edge == "north":
            m[0, :] = True
        elif edge == "south":
            m[-1, :] = True
        elif edge == "west":
            m[:, 0] = True
        elif edge == "east":
            m[:, -1] = True
        else:
            raise ValueError(edge)
        return m

    def distance_to_edge_m(self, edge: str) -> np.ndarray:
        r = np.arange(self.ny, dtype=np.float64)[:, None]
        c = np.arange(self.nx, dtype=np.float64)[None, :]
        if edge == "north":
            d = np.broadcast_to(r, (self.ny, self.nx))
        elif edge == "south":
            d = np.broadcast_to(self.ny - 1 - r, (self.ny, self.nx))
        elif edge == "west":
            d = np.broadcast_to(c, (self.ny, self.nx))
        elif edge == "east":
            d = np.broadcast_to(self.nx - 1 - c, (self.ny, self.nx))
        else:
            raise ValueError(edge)
        return np.array(d) * self.dx

    def edge_cells(self, edge: str) -> np.ndarray:
        """端のセル番号を、端に沿った順（北・南は西→東、東・西は北→南）で返す。"""
        if edge == "north":
            return np.arange(self.nx, dtype=np.int64)
        if edge == "south":
            return (self.ny - 1) * self.nx + np.arange(self.nx, dtype=np.int64)
        if edge == "west":
            return np.arange(self.ny, dtype=np.int64) * self.nx
        if edge == "east":
            return np.arange(self.ny, dtype=np.int64) * self.nx + self.nx - 1
        raise ValueError(edge)


def region_label(row: float, col: float, grid: Grid) -> str:
    """位置を「北西部」「中央」などの方角で表す（3×3 の区分）。"""
    v = ["北", "", "南"][min(2, int(3 * row / grid.ny))]
    h = ["西", "", "東"][min(2, int(3 * col / grid.nx))]
    if not v and not h:
        return "中央部"
    return f"{v}{h}部"
