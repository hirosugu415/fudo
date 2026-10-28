import numpy as np

from fudo import rng


def test_stream_independent_of_other_streams():
    a1 = rng.stream(1, "terrain").random(5)
    rng.stream(1, "new_layer").random(100)  # 新しいストリームを追加しても
    a2 = rng.stream(1, "terrain").random(5)
    assert np.array_equal(a1, a2)


def test_streams_differ_by_name_and_seed():
    assert not np.array_equal(rng.stream(1, "a").random(5), rng.stream(1, "b").random(5))
    assert not np.array_equal(rng.stream(1, "a").random(5), rng.stream(2, "a").random(5))


def test_run_stream_separate_from_world_stream():
    assert not np.array_equal(rng.stream(1, "x").random(5), rng.run_stream(1, "x").random(5))


def test_derived_run_seed_deterministic():
    assert rng.derive_run_seed(42) == rng.derive_run_seed(42)
    assert rng.derive_run_seed(42) != rng.derive_run_seed(43)
    assert 0 <= rng.derive_run_seed(42) < 2**rng.SEED_BITS
