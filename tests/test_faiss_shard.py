import numpy as np
import pytest

from driftdrag.index.faiss_shard import FaissShard


def test_add_and_search_returns_nearest_neighbor():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 8)).astype(np.float32)
    ids = [f"doc-{i}" for i in range(20)]

    shard = FaissShard(dim=8, index_type="flat")
    shard.add(embeddings, ids, model_version="v0")

    results = shard.search(embeddings[5], k=1)
    assert results[0][0] == "doc-5"


def test_mixing_model_versions_raises():
    rng = np.random.default_rng(0)
    shard = FaissShard(dim=4, index_type="flat")
    shard.add(rng.normal(size=(5, 4)).astype(np.float32), [f"d{i}" for i in range(5)], "v0")
    with pytest.raises(ValueError):
        shard.add(rng.normal(size=(5, 4)).astype(np.float32), [f"e{i}" for i in range(5)], "v1")


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(10, 6)).astype(np.float32)
    ids = [f"doc-{i}" for i in range(10)]

    shard = FaissShard(dim=6, index_type="flat")
    shard.add(embeddings, ids, model_version="v0")
    shard.save(tmp_path / "shard0")

    loaded = FaissShard.load(tmp_path / "shard0")
    assert loaded.model_version == "v0"
    assert loaded.ids == ids
    results = loaded.search(embeddings[3], k=1)
    assert results[0][0] == "doc-3"
