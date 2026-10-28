import pytest

from fudo.world.generate import generate_world, load_world

WORLD_SEED = 7


@pytest.fixture(scope="session")
def world_dir(tmp_path_factory):
    return generate_world(WORLD_SEED, out_root=tmp_path_factory.mktemp("worlds"), reuse=False, log=lambda *a: None)


@pytest.fixture(scope="session")
def world(world_dir):
    return load_world(world_dir)
