import fakeredis
import pytest

from driftdrag.ledger.drift_ledger import DriftLedger


@pytest.fixture
def ledger(tmp_path):
    client = fakeredis.FakeStrictRedis()
    return DriftLedger(client, history_path=tmp_path / "ledger_history.jsonl")


def test_get_missing_returns_none(ledger):
    assert ledger.get("shard-0", "v0", "v1") is None


def test_update_drift_vector_roundtrip(ledger):
    ledger.update_drift_vector("shard-0", "v0", "v1", [0.1, 0.2, 0.3])
    record = ledger.get("shard-0", "v0", "v1")
    assert record.drift_vector == [0.1, 0.2, 0.3]
    assert record.shard_id == "shard-0"


def test_per_cluster_drift_vectors(ledger):
    ledger.update_drift_vector("shard-0", "v0", "v1", [1.0, 0.0], cluster_id="0")
    ledger.update_drift_vector("shard-0", "v0", "v1", [0.0, 1.0], cluster_id="1")
    record = ledger.get("shard-0", "v0", "v1")
    assert record.per_cluster_drift == {"0": [1.0, 0.0], "1": [0.0, 1.0]}


def test_update_drift_scores(ledger):
    ledger.update_drift_scores("shard-0", "v0", "v1", anchor_residual=0.05, data_drift_score=0.2)
    record = ledger.get("shard-0", "v0", "v1")
    assert record.anchor_residual == 0.05
    assert record.data_drift_score == 0.2


def test_mark_reembedded_clears_drift_state(ledger):
    ledger.update_drift_vector("shard-0", "v0", "v1", [1.0, 2.0])
    ledger.mark_reembedded("shard-0", "v0", "v1", "v1")
    record = ledger.get("shard-0", "v0", "v1")
    assert record.drift_vector is None
    assert record.per_cluster_drift == {}
    assert record.last_reembed_version == "v1"


def test_all_shards_lists_every_record(ledger):
    ledger.update_drift_vector("shard-0", "v0", "v1", [1.0])
    ledger.update_drift_vector("shard-1", "v0", "v1", [2.0])
    records = ledger.all_shards()
    assert {r.shard_id for r in records} == {"shard-0", "shard-1"}


def test_export_history_parquet(ledger, tmp_path):
    ledger.update_drift_vector("shard-0", "v0", "v1", [1.0, 2.0])
    ledger.update_drift_scores("shard-0", "v0", "v1", anchor_residual=0.1)
    out = ledger.export_history_parquet(tmp_path / "history.parquet")
    import pandas as pd

    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "shard_id" in df.columns
