"""Drift State Ledger (Stage 1).

A distributed, append-only ledger recording, per shard and per model
transition (v_from -> v_to):
  - the estimated drift vector(s) (global / per-cluster)
  - anchor-set residual statistics
  - data-drift statistics
  - last re-embed timestamp and version

Backed by any redis-py-compatible client (real Redis/etcd-fronting proxy in
production, ``fakeredis`` in unit tests) so the same code runs unmodified
whether the ledger lives on one VM or is a dedicated shared service reachable
by every worker VM. History is additionally appended to a local/Parquet-on-
object-store log for offline analytics (Section 3 ablations read this, not
the live KV).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _key(shard_id: str, v_from: str, v_to: str) -> str:
    return f"ledger:{shard_id}:{v_from}->{v_to}"


@dataclass
class DriftRecord:
    shard_id: str
    v_from: str
    v_to: str
    drift_vector: list[float] | None = None
    per_cluster_drift: dict[str, list[float]] = field(default_factory=dict)
    anchor_residual: float | None = None
    data_drift_score: float | None = None
    last_reembed_ts: float | None = None
    last_reembed_version: str | None = None
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, blob: str) -> "DriftRecord":
        return cls(**json.loads(blob))


class DriftLedger:
    """Thin client over a Redis-compatible KV store plus a Parquet history log.

    Parameters
    ----------
    client:
        A redis-py (or fakeredis) client instance. Injected rather than
        constructed here so the same class serves unit tests, single-VM dev,
        and a multi-VM deployment where the ledger is a dedicated Redis VM.
    history_path:
        Local path (or mounted object-store path, e.g. an s3fs/minio mount)
        where the append-only JSONL history is written for offline analytics
        and Parquet export.
    """

    def __init__(self, client, history_path: str | Path = "data/lake/ledger_history.jsonl"):
        self.client = client
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, shard_id: str, v_from: str, v_to: str) -> Optional[DriftRecord]:
        blob = self.client.get(_key(shard_id, v_from, v_to))
        if blob is None:
            return None
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8")
        return DriftRecord.from_json(blob)

    def put(self, record: DriftRecord) -> None:
        record.updated_at = time.time()
        self.client.set(_key(record.shard_id, record.v_from, record.v_to), record.to_json())
        self._append_history(record)

    def update_drift_vector(
        self,
        shard_id: str,
        v_from: str,
        v_to: str,
        drift_vector: list[float],
        cluster_id: str | None = None,
    ) -> DriftRecord:
        record = self.get(shard_id, v_from, v_to) or DriftRecord(shard_id, v_from, v_to)
        if cluster_id is None:
            record.drift_vector = drift_vector
        else:
            record.per_cluster_drift[cluster_id] = drift_vector
        self.put(record)
        return record

    def update_drift_scores(
        self,
        shard_id: str,
        v_from: str,
        v_to: str,
        anchor_residual: float | None = None,
        data_drift_score: float | None = None,
    ) -> DriftRecord:
        record = self.get(shard_id, v_from, v_to) or DriftRecord(shard_id, v_from, v_to)
        if anchor_residual is not None:
            record.anchor_residual = anchor_residual
        if data_drift_score is not None:
            record.data_drift_score = data_drift_score
        self.put(record)
        return record

    def mark_reembedded(self, shard_id: str, v_from: str, v_to: str, version: str) -> DriftRecord:
        record = self.get(shard_id, v_from, v_to) or DriftRecord(shard_id, v_from, v_to)
        record.last_reembed_ts = time.time()
        record.last_reembed_version = version
        record.drift_vector = None
        record.per_cluster_drift = {}
        self.put(record)
        return record

    def all_shards(self, pattern: str = "ledger:*") -> list[DriftRecord]:
        records = []
        for key in self.client.keys(pattern):
            blob = self.client.get(key)
            if blob is None:
                continue
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8")
            records.append(DriftRecord.from_json(blob))
        return records

    def _append_history(self, record: DriftRecord) -> None:
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def export_history_parquet(self, out_path: str | Path) -> Path:
        """Flush the JSONL history log to Parquet for E1-E6 offline analysis."""
        import pandas as pd

        if not self.history_path.exists():
            raise FileNotFoundError(f"no ledger history at {self.history_path}")
        rows = [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line]
        df = pd.json_normalize(rows)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path


def connect(host: str = "localhost", port: int = 6379, db: int = 0, use_fake: bool = False):
    """Factory used by jobs/scripts so swapping real Redis for fakeredis (CI,
    local smoke tests without Docker) is a single config flag, not a code
    change.
    """
    if use_fake:
        import fakeredis

        return fakeredis.FakeStrictRedis(host=host, port=port, db=db)
    import redis

    return redis.Redis(host=host, port=port, db=db)
