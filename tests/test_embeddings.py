import numpy as np

from driftdrag.ingest.embeddings import HashingDriftBackend


def test_embed_is_deterministic_across_calls():
    backend = HashingDriftBackend(dim=32)
    a = backend.embed(["hello world"], model_version="v0")
    b = backend.embed(["hello world"], model_version="v0")
    assert np.allclose(a, b)


def test_embed_differs_across_model_versions():
    backend = HashingDriftBackend(dim=32)
    v0 = backend.embed(["hello world"], model_version="v0")
    v3 = backend.embed(["hello world"], model_version="v3")
    assert not np.allclose(v0, v3)


def test_drift_is_cumulative_across_versions():
    """v0->v1 and v1->v2 should each be a *small* step (bounded angle),
    matching the plan's cumulative-drift-vector premise (QDC), rather than
    an arbitrary jump.
    """
    backend = HashingDriftBackend(dim=16, drift_angle=0.1)
    text = ["a document"]
    v0 = backend.embed(text, "v0")[0]
    v1 = backend.embed(text, "v1")[0]
    v2 = backend.embed(text, "v2")[0]

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cos(v0, v1) > 0.9
    assert cos(v0, v2) > 0.8
    assert cos(v0, v2) <= cos(v0, v1) + 1e-9
