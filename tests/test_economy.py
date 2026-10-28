"""外部世界モデル（SPEC 第8章）：決定的で、計算の範囲を変えても同じ年の値は同じ。価格は需給で動く。"""

import numpy as np

from fudo.config import load_behavior_config
from fudo.society.economy import External

INFO = {
    "cities": [{"population": 150000.0}, {"population": 80000.0}],
    "population_growth_per_year": 0.01,
    "business_cycle_amplitude": 0.03,
}


def test_same_year_same_value_regardless_of_end():
    o = load_behavior_config().outside
    a = External(5, INFO, o, 1900).at(1870)
    b = External(5, INFO, o, 2150).at(1870)
    assert a.wage_yen_per_day == b.wage_yen_per_day and a.prices == b.prices
    assert np.array_equal(a.city_pop, b.city_pop) and a.interest == b.interest


def test_anchor_values_and_seed_dependence():
    o = load_behavior_config().outside
    e = External(5, INFO, o, 2000)
    y = e.at(int(o.anchor_year))
    assert y.productivity == 1.0
    assert np.allclose(y.city_pop, [150000.0, 80000.0])
    assert not np.array_equal(e.x, External(6, INFO, o, 2000).x)


def test_price_rises_in_booms_and_population_growth_slows():
    o = load_behavior_config().outside
    e = External(5, INFO, o, 2150)
    # 景気がよい年ほど鉱物の値段が高い（需要が増え、供給は遅れて追いつく）
    assert np.corrcoef(e.x, e.prices["coal"])[0, 1] > 0.3
    # 所得が上がると人口の伸びが鈍る（人口転換）
    g = np.diff(np.log(e.pop[:, 0]))
    i1900, i2100 = 1900 - e.origin, 2100 - e.origin
    assert g[i2100] < 0.5 * g[i1900]
